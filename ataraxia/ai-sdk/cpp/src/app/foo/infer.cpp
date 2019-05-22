#include "infer.hpp"

#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include "glog/logging.h"
#include "google/protobuf/io/zero_copy_stream.h"
#include "google/protobuf/text_format.h"

#include "common/archiver.hpp"
#include "common/errors.hpp"
#include "common/image.hpp"
#include "forward.hpp"
#include "framework/context.hpp"
#include "inference.hpp"
#include "proto/inference.pb.h"

struct CustomParams {
  CustomParams() {}
  ~CustomParams() {}

  std::string frontend;
  std::string backend;
  int channel;
  int width;
  int height;
};

using Config = tron::framework::Config<CustomParams>;
using Context = tron::framework::Context<CustomParams>;
using Handle = tron::framework::Handle<CustomParams>;
using inference::foo::ForwardRequest;
using inference::foo::ForwardResponse;
using tron::foo::Forward;
using tron::foo::Inference;
using tron::framework::ForwardWrap;

inline bool check_valid_box_pts(const int pts[4][2]) {
  if (pts[0][0] == pts[3][0] && pts[0][1] == pts[1][1] &&
      pts[1][0] == pts[2][0] && pts[2][1] == pts[3][1] &&
      pts[2][0] > pts[0][0] && pts[2][1] > pts[0][1]) {
    return true;
  }

  return false;
}

thread_local char *_Error_;
const char *QTGetLastError() { return _Error_; }

template <typename Archiver>
Archiver &operator&(Archiver &ar, CustomParams &p) {  // NOLINT
  ar.StartObject();
  ar.Member("width") & p.width;
  ar.Member("height") & p.height;
  ar.Member("channel") & p.channel;
  ar.Member("frontend") & p.frontend;
  ar.Member("backend") & p.backend;
  return ar.EndObject();
}

int QTPredCreate(const void *in_data, const int in_size,
                 PredictorContext *out) {
  auto ctx = tron::framework::CreateContext<CustomParams>();

  LOG(INFO) << "create start";
  Config config;
  ctx->ParseConfig(in_data, in_size,
                   std::vector<std::vector<std::string>>(),
                   nullptr,
                   &config);

  LOG(INFO) << "batch_size=" << config.batch_size
            << " in_shape=["
            << config.params.channel << ","
            << config.params.height << ","
            << config.params.width << "]";

  LOG(INFO) << "frontend=" << config.params.frontend
            << " backend=" << config.params.backend;
  ctx->AddForward<
      Forward,
      ForwardRequest,
      ForwardResponse>(std::vector<std::vector<char>>(),
                       {
                           config.batch_size,
                           config.params.channel,
                           config.params.height,
                           config.params.width,
                       },
                       0,
                       config.params.frontend,
                       config.params.backend,
                       "", 2);

  *out = ctx;
  LOG(INFO) << "create finished";
  return 0;
}

int QTPredHandle(PredictorContext ctx,
                 const void *, const int,
                 PredictorHandle *handle) {
  LOG(INFO) << "handle begin...";
  Context *c = reinterpret_cast<Context *>(ctx);
  auto h = tron::framework::CreateHandle<CustomParams>();
  auto frontend =
      h->AddForward<Forward,
                    ForwardRequest,
                    ForwardResponse>(c, c->config_.params.frontend);
  auto inference = std::make_shared<tron::foo::Inference>();
  inference->Setup({c->config_.batch_size,
                    c->config_.params.channel,
                    c->config_.params.height,
                    c->config_.params.width},
                   frontend);
  h->Setup<tron::foo::Inference>(c, inference);
  *handle = h;
  LOG(INFO) << "handle done.";
  return 0;
}

struct Response {
  Response() {}
  ~Response() {}

  double sum;
};

template <typename Archiver>
Archiver &operator&(Archiver &ar, Response &resp) {  // NOLINT
  ar.StartObject();
  ar.Member("sum") & resp.sum;
  return ar.EndObject();
}

int QTPredInference(PredictorHandle handle,
                    const void *in_data, const int in_size,
                    void **out_data, int *out_size) {
  LOG(INFO) << "netInference() start. " << in_size;

  Handle *h = reinterpret_cast<Handle *>(handle);

  inference::InferenceRequests requests;
  requests.ParseFromArray(in_data, in_size);

  std::vector<cv::Mat> im_mats;
  for (int i = 0; i < requests.requests_size(); i++) {
    auto request = requests.mutable_requests(i);
    auto req_data = request->mutable_data();
    if (!req_data->has_body() || req_data->mutable_body()->empty()) {
      LOG(WARNING) << "RequestData body is empty!";
      _Error_ = const_cast<char *>(tron::get_status_message(
          tron::tron_status_request_data_body_empty));
      return 1;
    }
    auto im_mat = tron::decode_image_buffer(*req_data->mutable_body());
    if (im_mat.empty()) {
      LOG(WARNING) << "OpenCV decode buffer error!";
      _Error_ = const_cast<char *>(tron::get_status_message(
          tron::tron_status_imdecode_error));
      return 1;
    }

    im_mats.push_back(im_mat);
  }

  LOG(INFO) << "predict begin. " << (h->GetInference<Inference>() == nullptr)
            << " " << im_mats.size();
  std::vector<float> rets;
  h->GetInference<Inference>()->Predict(im_mats, &rets);
  LOG(INFO) << "predict end";

  inference::InferenceResponses responses;
  for (std::size_t i = 0; i < rets.size(); i++) {
    auto *resp = responses.add_responses();
    {
      Response response;
      response.sum = rets[i];
      JsonWriter writer;
      writer &response;
      resp->set_result(writer.GetString());
    }
  }

  h->Return(responses, out_data, out_size);

  LOG(INFO) << "netInference() finished";
  return 0;
}

int QTPredFree(PredictorContext ctx) {
  Context *c = reinterpret_cast<Context *>(ctx);
  delete c;
  return 0;
}
