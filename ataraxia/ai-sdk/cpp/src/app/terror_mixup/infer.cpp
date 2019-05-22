#include "infer.hpp"
#include "common/errors.hpp"
#include "common/image.hpp"
#include "forward.hpp"
#include "framework/context.hpp"
#include "glog/logging.h"
#include "gsl/span"
#include "inference.hpp"

thread_local char *_Error_;
const char *QTGetLastError() { return _Error_; }

int QTPredCreate(const void *in_data, const int in_size,
                 PredictorContext *out) {
  auto ctx = tron::framework::CreateContext<tron::terror_mixup::CustomParams>();
  std::vector<std::vector<std::vector<char>>> net_para_datas;
  {
    // TODO(wkc): 这里容易坑，不使用 ParseConfig
    // 返回的config，框架去掉这个返回参数？
    tron::terror_mixup::Config _ignore;
    ctx->ParseConfig(in_data, in_size,
                     std::vector<std::vector<std::string>>{
                         tron::terror_mixup::forward_custom_file_list,
                         tron::terror_mixup::inference_custom_file_list},
                     &net_para_datas, &_ignore);
  }
  CHECK_EQ(net_para_datas[0].size(),
           tron::terror_mixup::forward_custom_file_list.size());
  CHECK_EQ(net_para_datas[1].size(),
           tron::terror_mixup::inference_custom_file_list.size());
  if (ctx->config_.params.frontend == "") {
    ctx->config_.params.frontend = "inproc://frontend";
  }
  if (ctx->config_.params.backend == "") {
    ctx->config_.params.backend = "inproc://backend";
  }
  LOG(INFO) << "batch_size:" << ctx->config_.batch_size
            << " frontend:" << ctx->config_.params.frontend
            << " backend:" << ctx->config_.params.backend
            << " gpu_id:" << ctx->config_.params.gpu_id;
  // TODO(wkc): inference 不支持传文件，workaround
  ctx->config_.params.files = {
      tron::terror_mixup::vector_char_to_string(net_para_datas[1][0]),
      tron::terror_mixup::vector_char_to_string(net_para_datas[1][1]),
      tron::terror_mixup::vector_char_to_string(net_para_datas[1][2]),
  };
  CHECK_EQ(ctx->config_.params.files.size(),
           tron::terror_mixup::inference_custom_file_list.size());
  std::vector<int> in_shape = {
      ctx->config_.batch_size,
      0,
      0,
      0,
  };
  const std::string name = "";
  const int num = 1;
  ctx->AddForward<
      tron::terror_mixup::Forward, tron::terror_mixup::ForwardRequest,
      tron::terror_mixup::ForwardResponse, tron::terror_mixup::Config>(
      net_para_datas[0], in_shape, ctx->config_.params.gpu_id,
      ctx->config_.params.frontend, ctx->config_.params.backend, name, num,
      ctx->config_);
  *out = ctx;
  LOG(INFO) << "create finished";
  return 0;
}

int QTPredHandle(PredictorContext ctx, const void *, const int,
                 PredictorHandle *handle) {
  tron::terror_mixup::Context *c =
      reinterpret_cast<tron::terror_mixup::Context *>(ctx);
  auto h = tron::framework::CreateHandle<tron::terror_mixup::CustomParams>();
  auto frontend = h->AddForward<tron::terror_mixup::Forward,
                                tron::terror_mixup::ForwardRequest,
                                tron::terror_mixup::ForwardResponse>(
      c, c->config_.params.frontend);
  CHECK_EQ(c->config_.params.files.size(),
           tron::terror_mixup::inference_custom_file_list.size());
  auto inference = std::make_shared<tron::terror_mixup::Inference>();
  const std::vector<int> in_shape = {c->config_.batch_size, 0, 0, 0};
  inference->Setup(in_shape, frontend, c->config_);
  h->Setup<tron::terror_mixup::Inference>(c, inference);
  *handle = h;
  return 0;
}

int QTPredInference(PredictorHandle handle, const void *in_data,
                    const int in_size, void **out_data, int *out_size) {
  LOG(INFO) << "netInference() start. in_size:" << in_size;
  tron::terror_mixup::Handle *h =
      reinterpret_cast<tron::terror_mixup::Handle *>(handle);
  inference::InferenceRequests requests;
  requests.ParseFromArray(in_data, in_size);
  std::vector<cv::Mat> im_mats;
  for (int i = 0; i < requests.requests_size(); i++) {
    auto request = requests.requests(i);
    auto req_data = request.data();
    if (!req_data.has_body() || req_data.body().empty()) {
      LOG(WARNING) << "RequestData body is empty!";
      _Error_ = const_cast<char *>(
          tron::get_status_message(tron::tron_status_request_data_body_empty));
      return 1;
    }
    auto im_mat = tron::decode_image_buffer(req_data.body());
    if (im_mat.empty()) {
      LOG(WARNING) << "OpenCV decode buffer error!";
      _Error_ = const_cast<char *>(
          tron::get_status_message(tron::tron_status_imdecode_error));
      return 1;
    }
    im_mats.push_back(im_mat);
  }

  LOG(INFO) << "predict begin. size:" << im_mats.size();
  std::vector<std::string> rets;
  h->GetInference<tron::terror_mixup::Inference>()->Predict(im_mats, &rets);
  inference::InferenceResponses responses;
  LOG(INFO) << "predict end";

  for (std::size_t i = 0; i < rets.size(); i++) {
    auto *resp = responses.add_responses();
    resp->set_result(rets[i]);
  }

  h->Return(responses, out_data, out_size);

  LOG(INFO) << "netInference() finished";
  return 0;
}

int QTPredFree(PredictorContext ctx) {
  tron::terror_mixup::Context *c =
      reinterpret_cast<tron::terror_mixup::Context *>(ctx);
  delete c;
  return 0;
}
