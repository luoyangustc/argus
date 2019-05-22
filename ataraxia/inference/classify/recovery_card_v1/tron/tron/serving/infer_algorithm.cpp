#include "infer_algorithm.hpp"

#include "infer_implement.hpp"

#include "common/helper.hpp"

using namespace Tron;

struct TronInferHandle {
  // Serving limit the maximum response buffer size to 4M bytes
  const int responses_max_buffer_size_ = 4 * 1024 * 1024;
  // gpu index
  int gpu_id_ = 0;
  // Used to return results whose scores larger than threshold for
  // classification
  float threshold_ = std::numeric_limits<float>::lowest();
  // Used to return the top k results for classification
  int top_k_ = -1;
  // Magic method for your task, change this to your algorithm implement
  Tron::Classification *method_ = nullptr;
};

// Function used to do the inference
int TronProcess(const cv::Mat &im_mat, TronInferHandle *handle,
                TronClassificationOutput *output) {
  if (handle->method_ == nullptr) {
    LOG(WARNING) << "Tron is uninitialized";
    return tron_status_method_nullptr;
  }

  int im_h = im_mat.rows, im_w = im_mat.cols;

  if (im_h <= 1 || im_w <= 1) {
    return tron_status_image_size_error;
  }

  std::vector<std::map<std::string, Tron::VecFloat>> Gscores;
  Tron::VecRectF rois{Tron::RectF(0, 0, im_w, im_h)};
  handle->method_->Predict(im_mat, rois, &Gscores);

  output->scores = Gscores[0].at("score");
  std::vector<std::string> labels;
  handle->method_->GetLabels(&labels);
  output->labels = labels;
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

/**
 * void initEnv(InitParams params,
 *              const int params_size,
 *              int* code,
 *              char** err);
 */
void initEnv(void *params, const int params_size, int *code, char **err) {
  // void implement
}

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
  if (document.HasMember("threshold")) {
    handle_->threshold_ = Tron::get_float(document, "threshold", -1);
  }
  if (document.HasMember("top_k")) {
    handle_->top_k_ = Tron::get_int(document, "top_k", -1);
  }

  // Initial network
  handle_->method_ = new Tron::Classification();
  handle_->method_->Setup(meta_net_param, {1}, handle_->gpu_id_);

  *code = tron_status_success;
  *err = const_cast<char *>(Tron::get_status_message(*code));

  LOG(INFO) << "Finished createNet!";

  return handle_;
}

/**
 * void netInference(void* ctx,
 *                   InferenceRequests requests,
 *                   const int requests_size,
 *                   int* code,
 *                   char** err,
 *                   InferenceResponses ret,
 *                   int* ret_size);
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
        TronClassificationOutput classification_output = {};
        int status = TronProcess(im_mat, ptr, &classification_output);
        if (status == tron_status_success) {
          // Transform result struct to json string
          const auto &result_json_str = Tron::get_classify_json(
              classification_output, ptr->top_k_, ptr->threshold_);
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
