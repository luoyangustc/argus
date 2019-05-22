#include "infer.hpp"

#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/opencv.hpp>
#include "glog/logging.h"
#include "google/protobuf/io/zero_copy_stream.h"
#include "google/protobuf/text_format.h"

#include "common/archiver.hpp"
#include "common/errors.hpp"
#include "common/image.hpp"
#include "common/json.hpp"
#include "common/md5.hpp"
#include "framework/context.hpp"
#include "proto/inference.pb.h"

using namespace tron;  // NOLINT

struct CustomParams {
  CustomParams() {}
  ~CustomParams() {}

  std::string frontend;
  std::string backend;

  int gpu_id;

  std::map<int, std::pair<std::string, float>> det_labels;
  std::map<int, std::string> fine_labels;
};

using Config = tron::framework::Config<CustomParams>;
using Context = tron::framework::Context<CustomParams>;
using Handle = tron::framework::Handle<CustomParams>;
using inference::wa::ForwardRequest;
using inference::wa::ForwardResponse;
using tron::framework::ForwardWrap;
using tron::wa::Forward;

template <typename Archiver>
Archiver &operator&(Archiver &ar, CustomParams &p) {  // NOLINT
  ar.StartObject();
  ar.Member("frontend") & p.frontend;
  ar.Member("backend") & p.backend;

  ar.Member("gpu_id") & p.gpu_id;

  return ar.EndObject();
}

template <typename Archiver>
Archiver &operator&(Archiver &ar, tron::wa::DetectionResult &p) {  // NOLINT
  ar.StartObject();
  ar.Member("index") & p.index;
  ar.Member("score") & p.score;
  ar.Member("class") & p.clas;
  ar.Member("pts");
  size_t i = 4;
  ar.StartArray(&i);
  size_t j1 = 2;
  ar.StartArray(&j1);
  ar &p.pts[0];
  ar &p.pts[1];
  ar.EndArray();
  size_t j2 = 2;
  ar.StartArray(&j2);
  ar &p.pts[2];
  ar &p.pts[1];
  ar.EndArray();
  size_t j3 = 2;
  ar.StartArray(&j3);
  ar &p.pts[2];
  ar &p.pts[3];
  ar.EndArray();
  size_t j4 = 2;
  ar.StartArray(&j4);
  ar &p.pts[0];
  ar &p.pts[3];
  ar.EndArray();
  ar.EndArray();
  return ar.EndObject();
}
template <typename Archiver>
Archiver &operator&(Archiver &ar, tron::wa::ConfidenceResult &p) {  // NOLINT
  ar.StartObject();
  ar.Member("index") & p.index;
  ar.Member("score") & p.score;
  ar.Member("class") & p.clas;
  return ar.EndObject();
}

// {
//   "classify" : {
//     "confidences" : [
//       {
//         "class" : "normal",
//         "index" : 47,
//         "score" : 0.9989687204360962
//       }
//     ]
//   },
//   "detection" : [
//     {
//       "class" : "guns_true",
//       "index" : 4,
//       "pts" : [ [ 2, 96 ], [ 500, 96 ], [ 500, 419 ], [ 2, 419 ] ],
//       "score" : 0.8570531010627747
//     }
//   ]
// }
template <typename Archiver>
Archiver &operator&(Archiver &ar, tron::wa::Result &p) {  // NOLINT
  ar.StartObject();
  ar.Member("classify");
  ar.StartObject();
  ar.Member("confidences");
  size_t n = p.confidences.size();
  ar.StartArray(&n);
  for (std::size_t i = 0; i < p.confidences.size(); i++) {
    ar &p.confidences.at(i);
  }
  ar.EndArray();
  ar.EndObject();
  ar.Member("detection");
  size_t m = p.detections.size();
  ar.StartArray(&m);
  for (std::size_t i = 0; i < p.detections.size(); i++) {
    ar &p.detections.at(i);
  }
  ar.EndArray();
  return ar.EndObject();
}

thread_local char *_Error_;
const char *QTGetLastError() { return _Error_; }

int QTPredCreate(const void *in_data, const int in_size,
                 PredictorContext *out) {
  auto ctx = tron::framework::CreateContext<CustomParams>();

  LOG(INFO) << "facex-feature-tron: createNet() start";

  Config config;
  std::vector<std::vector<std::vector<char>>> net_param_datas;
  ctx->ParseConfig(in_data, in_size,
                   std::vector<std::vector<std::string>>{
                       {
                           "fine_weight.bin",
                           "det_weight.bin",
                       },
                       {
                           "fine_labels.csv",
                           "det_labels.csv",
                       },
                   },
                   &net_param_datas,
                   &config);

  {
    std::istringstream ss(std::string(net_param_datas[1][1].begin(),
                                      net_param_datas[1][1].end()));
    bool firstLineFlag = true;
    for (std::string line; std::getline(ss, line);) {
      if (firstLineFlag) {
        firstLineFlag = false;
        continue;
      }
      std::string::size_type i1(line.find_first_of(','));
      std::string index(line, 0, i1);
      std::string::size_type i2(line.find_first_of(',', i1 + 1));
      std::string label(line, i1 + 1, i2 - i1 - 1);
      std::string threshold(line, i2 + 1);
      ctx->config_.params.det_labels[std::stoi(index)] =
          std::pair<std::string, float>(label, std::stof(threshold));
    }
  }
  {
    std::istringstream ss(std::string(net_param_datas[1][0].begin(),
                                      net_param_datas[1][0].end()));
    bool firstLineFlag = true;
    for (std::string line; std::getline(ss, line);) {
      if (firstLineFlag) {
        firstLineFlag = false;
        continue;
      }
      std::string::size_type i1(line.find_first_of(','));
      std::string index(line, 0, i1);
      std::string label(line, i1 + 1);
      ctx->config_.params.fine_labels[std::stoi(index)] = label;
    }
  }
  config = ctx->config_;

  ctx->AddForward<
      Forward,
      ForwardRequest,
      ForwardResponse,
      tron::wa::ForwardConfig>(net_param_datas[0],
                               {config.batch_size, 3, 512, 512},
                               config.params.gpu_id,
                               config.params.frontend,
                               config.params.backend,
                               "det", 1,
                               tron::wa::ForwardConfig(1, 0.1, 0.1));

  *out = ctx;
  LOG(INFO) << "facex-feature-tron: createNet() finished";
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
  auto inference = std::make_shared<tron::wa::Inference>();
  inference->Setup({c->config_.batch_size, 3, 512, 512}, frontend,
                   tron::wa::InferenceConfig(c->config_.params.det_labels,
                                             c->config_.params.fine_labels));

  h->Setup<tron::wa::Inference>(c, inference);
  *handle = h;
  LOG(INFO) << "handle done.";
  return 0;
}

int QTPredInference(PredictorHandle handle,
                    const void *in_data, const int in_size,
                    void **out_data, int *out_size) {
  LOG(INFO) << "facex-feature-tron: netInference() start";

  Handle *h_ = reinterpret_cast<Handle *>(handle);

  // Parse InferenceRequests data
  inference::InferenceRequests _requests;
  _requests.ParseFromArray(in_data, in_size);
  inference::InferenceResponses responses;

  for (int i = 0; i < _requests.requests_size(); i++) {
    auto request = _requests.requests(i);
    auto *resp = responses.add_responses();

    std::vector<cv::Point2d> cv_faces_points;

    const auto &req_data = request.data();
    if (!req_data.has_body() || req_data.body().empty()) {
      LOG(WARNING) << "RequestData body is empty!";
      auto code = tron_status_request_data_body_empty;
      _Error_ = const_cast<char *>(get_status_message(code));
      return code;
    }
    auto im_mat = tron::decode_image_buffer(req_data.body());
    if (im_mat.empty()) {
      LOG(WARNING) << "OpenCV decode buffer error!";
      auto code = tron_status_imdecode_error;
      _Error_ = const_cast<char *>(get_status_message(code));
      return code;
    }

    LOG(INFO) << "begin feature";
    std::vector<tron::wa::Result> results;
    h_->GetInference<tron::wa::Inference>()->Predict({im_mat}, &results);
    LOG(INFO) << "end feature";

    JsonWriter writer;
    writer &results[0];
    resp->set_result(writer.GetString());
  }  // for

  h_->Return(responses, out_data, out_size);
  LOG(INFO) << "facex-feature-tron: netInference() finished";
  return 0;
}

int QTPredFree(PredictorContext ctx) {
  Context *c = reinterpret_cast<Context *>(ctx);
  delete c;
  return 0;
}
