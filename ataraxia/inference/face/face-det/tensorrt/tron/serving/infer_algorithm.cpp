#include "infer_algorithm.hpp"
#include "common/helper.hpp"
using namespace std;
#include <opencv2/opencv.hpp>
#include "../../third_party/mix/include/Net.hpp"
#include <vector>
#include <string>
using namespace cv;
using namespace Tron;
using namespace Shadow;

inline const cv::Mat decode_image_buffer(const std::string &im_data) {
  std::vector<char> im_buffer(im_data.begin(), im_data.end());
  try {
    cv::Mat im = cv::imdecode(cv::Mat(im_buffer), 1);
    if(im.channels()==4){
      cv::Mat bgr;
      cv::cvtColor(im,bgr,cv::COLOR_BGRA2BGR);
      return bgr;
    }else{
      return im;
    }
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
  vector<vector<char>> engin;
  vector<int>  engin_size;
  string custom_params;
  //parse create params
  if(!Tron::parse_create_params(params, params_size, custom_params, engin, engin_size)){
    *err  = const_cast<char *>(Tron::get_status_message(tron_status_parse_model_error));
    *code = tron_error_round(tron_status_parse_model_error);
    delete handle_;
    return nullptr;
  }
  //parse custom params
  if(!Tron::parse_custom_params(custom_params,handle_)){
    *err  = const_cast<char *>(Tron::get_status_message(tron_status_parse_custom_error));
    *code = tron_error_round(tron_status_parse_custom_error);
    delete handle_;
    return nullptr;
  }
  LOG(INFO) << "Handle custom params ";
  LOG(INFO) << "gpuid: "<<handle_->gpu_id_;
  LOG(INFO) << "model num: "<<handle_->model_num_;  
  handle_->net_ = createNet(handle_->model_num_);
  *code = handle_->net_->init(handle_->gpu_id_,engin,engin_size);
  if(*code != ShadowStatus::shadow_status_success){
    *err = const_cast<char *>(Tron::get_status_message(*code));
    *code = tron_error_round(*code);
    delete handle_;
    return nullptr;
  }  

  *err = const_cast<char *>(Tron::get_status_message(tron_status_success));
  *code = tron_status_success;

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
   std::vector<cv::Mat> imgs_mat_;
   std::vector<std::string> imgs_attribute_;
 
   //normal processing
   for (int n = 0; n < requests_.requests_size(); ++n) {
     const auto &req_data = requests_.requests(n).data();
     auto *res = responses_.add_responses();
     res->set_code(0);
     if (req_data.has_body() && !req_data.body().empty()) {
       const auto &im_mat = decode_image_buffer(req_data.body());
       const std::string im_attr = req_data.attribute();
       if (!im_mat.empty()) {
         if (im_mat.rows <= 1 || im_mat.cols <= 1) {//image_size Error
           res->set_code(tron_error_round(tron_status_image_size_error));
           res->set_message(get_status_message(tron_status_image_size_error));
           continue;
         }else{
           imgs_mat_.push_back(im_mat);
           imgs_attribute_.push_back(im_attr);
           continue;
         }
       }else{ //decode Error
         res->set_code(tron_error_round(tron_status_imdecode_error));
         res->set_message(get_status_message(tron_status_imdecode_error));
         continue;
       }
     }else{//request_data_empty Error
       res->set_code(tron_error_round(tron_status_request_data_body_empty));
       res->set_message(get_status_message(tron_status_request_data_body_empty));
       continue;
     }
   }

   //Tron::Timer inferTimer;
   //inferTimer.start();
   vector<string> result_json_str;
   *code = ptr->net_->predict(imgs_mat_,imgs_attribute_,result_json_str);
   if(*code != ShadowStatus::shadow_status_success){
     *err = const_cast<char *>(Tron::get_status_message(*code));
     *code = tron_error_round(*code);
     return;
   }
   //LOG(INFO)<<inferTimer.get_millisecond()/requests_.requests_size()<< "MS";

   //set result_json_str
   int index=0;
   for (int n = 0; n < requests_.requests_size(); ++n) {
     if(responses_.responses(n).code() == 0){
       responses_.mutable_responses(n)->set_code(tron_status_success);
       responses_.mutable_responses(n)->set_message(get_status_message(tron_status_success));
       responses_.mutable_responses(n)->set_result(result_json_str[index++]);
     }
   }

   // Check responses buffer size must be not larger than 4M bytes
   int responses_size = responses_.ByteSize();
   if (responses_size <= ptr->responses_max_buffer_size_) {
     responses_.SerializeToArray(ret, responses_size);
     *ret_size = responses_size;
     *err = const_cast<char *>(Tron::get_status_message(tron_status_success));
     *code = tron_status_success;
   } else {
     LOG(WARNING) << "Responses buffer size request for "
                  << responses_size / (1024 * 1024) << "MB";
     *err = const_cast<char *>(Tron::get_status_message(tron_status_response_buffer_not_enough));
     *code = tron_error_round(tron_status_response_buffer_not_enough);
   }
 }
