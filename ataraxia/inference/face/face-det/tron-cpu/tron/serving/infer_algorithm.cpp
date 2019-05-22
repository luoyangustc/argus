#include "infer_algorithm.hpp"

#include "infer_implement.hpp"

#include "common/helper.hpp"

using namespace Tron;

struct TronInferHandle {
  // Serving limit the maximum response buffer size to 4M bytes
  const int responses_max_buffer_size_ = 4 * 1024 * 1024;
  // gpu index
  int gpu_id_ = 0;
  // Magic method for your task, change this to your algorithm implement
  Tron::ShadowDetectionRefineDet *method_ = nullptr;
  std::vector<std::string> labels_;
};

int TronProcess(const cv::Mat &im_mat, TronInferHandle *handle,
                TronDetectionOutput *output) {
  if (handle->method_ == nullptr) {
    LOG(WARNING) << "Tron is uninitialized";
    return tron_status_method_nullptr;
  }

  int im_h = im_mat.rows, im_w = im_mat.cols;

  if (im_h <= 1 || im_w <= 1) {
    return tron_status_image_size_error;
  }

  // detect out
  std::vector<Tron::VecBoxF> Gboxes;
  std::vector<std::vector<Tron::VecPointF>> Gpoints;
  Tron::VecRectF rois{Tron::RectF(0, 0, im_w, im_h)};
  handle->method_->Predict(im_mat, rois, &Gboxes, &Gpoints);

  const auto &boxes = Gboxes[0];
  output->objects.clear();
  for (const auto &box : boxes) {
    TronRectangle rect = {};
    rect.xmin = static_cast<int>(box.xmin);
    rect.xmax = static_cast<int>(box.xmax);
    rect.ymin = static_cast<int>(box.ymin);
    rect.ymax = static_cast<int>(box.ymax);
    rect.id = box.label;
    rect.score = box.score;
    if (box.label >= 0 && box.label < handle->labels_.size()) {
      rect.label = handle->labels_[box.label];
    } else {
      rect.label = "";
    }
    output->objects.push_back(rect);
  }
  // done
  return tron_status_success;
}

inline const cv::Mat decode_image_buffer(const std::string &im_data) {
  std::vector<char> im_buffer(im_data.begin(), im_data.end());
  try {
    return cv::imdecode(cv::Mat(im_buffer), 1);
  } catch (cv::Exception &e) {
    LOG(WARNING) << e.msg;
    return cv::Mat();
  }
}

void initEnv(void *params, const int params_size, int *code, char **err) {}

/**
 * void* createNet(CreateParams params,
 *                 const int params_size,
 *                 int* code,
 *                 char** err);
 */
void *createNet(void *params, const int params_size, int *code, char **err) {
  LOG(INFO) << "Starting createNet!";

  // Check some pointers not be nullptr
  CHECK_NOTNULL(code) << "code is nullptr";
  CHECK_NOTNULL(err) << "err is nullptr";

  // Initial a handle
  auto *handle_ = new TronInferHandle();

  // Parse CreateParams data
  std::string custom_params;
  tron::MetaNetParam meta_net_param;
  bool success = Tron::parse_create_params(params, params_size, &custom_params,
                                           &meta_net_param);

  if (!success) {
    *code = tron_status_parse_model_error;
    *err = const_cast<char *>(Tron::get_status_message(*code));
    return nullptr;
  }

  // Parse some useful params
  const auto &document = Tron::get_document(custom_params);
  if (document.HasMember("gpu_id")) {
    handle_->gpu_id_ = Tron::get_int(document, "gpu_id", 0);
  }

  handle_->method_ = new Tron::ShadowDetectionRefineDet();
  handle_->method_->Setup(meta_net_param, {1}, handle_->gpu_id_);

  *code = tron_status_success;
  *err = const_cast<char *>(Tron::get_status_message(*code));

  LOG(INFO) << "Finished createNet!";

  return handle_;
}

// void *netPreprocess(const void *ctx, void *request, const int request_size,
//                    int *code, char **err, int *ret_size) {
//  return nullptr;
//}

/**
 * InferenceResponses netInference(void* ctx,
 *                                 InferenceRequests requests,
 *                                 const int requests_size,
 *                                 int* code,
 *                                 char** err,
 *                                 int* ret_size);
 */
void netInference(const void *ctx, void *requests, const int requests_size,
                  int *code, char **err, void *ret, int *ret_size) {
  // Check some pointers not be nullptr
  CHECK_NOTNULL(code) << "code is nullptr";
  CHECK_NOTNULL(err) << "err is nullptr";
  CHECK_NOTNULL(ret) << "ret is nullptr";
  CHECK_NOTNULL(ret_size) << "ret_size is nullptr";

  // Parse InferenceRequests data
  auto *ptr = reinterpret_cast<TronInferHandle *>(const_cast<void *>(ctx));
  inference::InferenceRequests requests_;
  requests_.ParseFromArray(requests, requests_size);

  inference::InferenceResponses responses_;
  for (int n = 0; n < requests_.requests_size(); ++n) {
    const auto &req_data = requests_.requests(n).data();
    auto *res = responses_.add_responses();
    if (req_data.has_body() && !req_data.body().empty()) {
      const auto &im_mat = decode_image_buffer(req_data.body());
      if (!im_mat.empty()) {
        // Process image, preprocess, network forward, post postprocess
        TronDetectionOutput detection_output = {};
        int status = TronProcess(im_mat, ptr, &detection_output);
        if (status == tron_status_success) {
          // Transform result struct to json string
          const auto &result_json_str = Tron::get_detect_json(detection_output);
          res->set_code(status);
          res->set_message(Tron::get_status_message(status));
          res->set_result(result_json_str);
        } else {
          *code = status;
          *err = const_cast<char *>(Tron::get_status_message(*code));
          return;
        }
      } else {
        LOG(WARNING) << "OpenCV decode buffer error!";
        *code = tron_status_imdecode_error;
        *err = const_cast<char *>(Tron::get_status_message(*code));
        return;
      }
    } else {
      LOG(WARNING) << "RequestData body is empty!";
      *code = tron_status_request_data_body_empty;
      *err = const_cast<char *>(Tron::get_status_message(*code));
      return;
    }
  }

  // Check responses buffer size must be not larger than 4M bytes
  int responses_size = responses_.ByteSize();
  if (responses_size <= ptr->responses_max_buffer_size_) {
    responses_.SerializeToArray(ret, responses_size);
    *ret_size = responses_size;
    *code = tron_status_success;
    *err = const_cast<char *>(Tron::get_status_message(*code));
  } else {
    LOG(WARNING) << "Responses buffer size request for "
                 << responses_size / (1024 * 1024) << "MB";
    *code = tron_status_response_buffer_not_enough;
    *err = const_cast<char *>(Tron::get_status_message(*code));
  }
}
