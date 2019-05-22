#include "infer_algorithm.hpp"
#include "infer_implement.hpp"
#include "proto/tron.pb.h"
#include "common/helper.hpp"

using namespace Tron;

struct TronInferHandle {
  // Serving limit the maximum response buffer size to 4M bytes
  const int responses_max_buffer_size_ = 4 * 1024 * 1024;
  // gpu index
  int gpu_id_ = 0;
  // Magic method for your task, change this to your algorithm implement
  Tron::MobileNetV2 *method_ = nullptr;

  int batch_size=MAX_BATCH_SIZE;

};

int TronProcess(const std::vector<Tron::TronMatRect>& im_mat_rect,//const cv::Mat &im_mat, const BoxF& face_rect_output,
                TronInferHandle *handle,
                TronLandmarkOutput* GLandmarkAspects) {
  if (handle->method_ == nullptr) {
    LOG(INFO) << "Tron is uninitialized";
    return tron_status_method_nullptr;
  }
  const int size=im_mat_rect.size();
  Tron::VecRectF rois;
  for(int i=0;i<size;++i){
      const int im_h = im_mat_rect[i].im_mat.rows, im_w = im_mat_rect[i].im_mat.cols;
      if (im_h <= 1 || im_w <= 1) {
        return tron_status_image_size_error;
      }
      rois.push_back(RectF(0, 0, im_w, im_h));
   }

  handle->method_->Predict(im_mat_rect,rois,GLandmarkAspects);
  return tron_status_success;
}

inline const cv::Mat decode_image_buffer(const std::string &im_data) {
  std::vector<char> im_buffer(im_data.begin(), im_data.end());
  try {
    return cv::imdecode(cv::Mat(im_buffer), 1);
  } catch (cv::Exception &e) {
    LOG(INFO) << e.msg;
    return cv::Mat();
  }
}


inline void expand_box(const BoxF& face_box_input,BoxF& face_box_output,
                       const cv::Size& im_size){
    const float center_x=(face_box_input.xmax+face_box_input.xmin)/2.;
    const float center_y=(face_box_input.ymax+face_box_input.ymin)/2.;
    const float width=face_box_input.xmax-face_box_input.xmin;
    const float height=face_box_input.ymax-face_box_input.ymin;
    const float scale=1.5;
    const float size=(width+height)/2. * scale;

    face_box_output.xmin=(center_x - 0.5 * size > 0)?(center_x - 0.5 * size):0;
    face_box_output.xmax=(center_x + 0.5 * size < im_size.width)?(center_x + 0.5 * size):im_size.width-1;
    face_box_output.ymin=(center_y - 0.5 * size > 0)?(center_y - 0.5 * size):0;
    face_box_output.ymax=(center_y + 0.5 * size < im_size.height) ? (center_y + 0.5 * size):im_size.height-1;
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

  handle_->method_ = new Tron::MobileNetV2();
  handle_->method_->Setup(meta_net_param, {handle_->batch_size}, handle_->gpu_id_);

  *code = tron_status_success;
  *err = const_cast<char *>(Tron::get_status_message(*code));

  LOG(INFO) << "Finished createNet!";

  return handle_;
}

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
   /*暂定1张图多人脸模式，后续再扩展成多张图多人脸模式:requests_.requests_size()==1*/
   for (int n = 0; n < requests_.requests_size(); ++n) {
      const auto &req_data = requests_.requests(n).data();
      auto *res = responses_.add_responses();
      if (req_data.has_body() && !req_data.body().empty()) {
            const auto &im_mat = decode_image_buffer(req_data.body());
            if (!im_mat.empty()) {
	      const cv::Size im_size=cv::Size(im_mat.cols,im_mat.rows);
              // Parse request face boxes from attribute
              VecBoxF face_boxes;
              if(!Tron::parse_request_boxes(req_data.attribute(),&face_boxes,im_size)){
                *code = tron_status_request_data_attribute_error;
                *err = const_cast<char *>(Tron::get_status_message(*code));
                return;
              }
              // Process image, preprocess, network forward, post postprocess  
              std::vector<Tron::TronMatRect> matRects;
              if(face_boxes.size()<1){
                LOG(WARNING) << "Attribute body had less than one face!";
                *code = tron_status_request_data_attribute_error;
                *err = const_cast<char *>(Tron::get_status_message(*code));
                return;
              }
              for(int iter=0;iter<face_boxes.size();iter++)
              {
                BoxF face_rect_output;
                expand_box(face_boxes[iter],face_rect_output,im_size);
                cv::Rect rect(face_rect_output.xmin,face_rect_output.ymin,1-face_rect_output.xmin+face_rect_output.xmax,1-face_rect_output.ymin+face_rect_output.ymax);
                Tron::TronMatRect FaceMat;
                FaceMat.im_mat=im_mat(rect);
                //im_face.copyTo(FaceMat.im_mat);
                FaceMat.face_rect_output.xmin=face_rect_output.xmin;
                FaceMat.face_rect_output.xmax=face_rect_output.xmax;
                FaceMat.face_rect_output.ymin=face_rect_output.ymin;
                FaceMat.face_rect_output.ymax=face_rect_output.ymax;
                matRects.push_back(FaceMat);
              }
              TronLandmarkOutput GLandmarkAspects={};
              int status = TronProcess(matRects, ptr, &GLandmarkAspects);
              if (status == tron_status_success) {
                // Transform result struct to json string
                const auto &result_json_str = Tron::get_landmark_json(GLandmarkAspects);
                res->set_code(status);
                res->set_message(Tron::get_status_message(status));
                res->set_result(result_json_str);
              } else {
                *code = status;
                *err = const_cast<char *>(Tron::get_status_message(*code));
                return;
              }

            }else{
              LOG(WARNING) << "OpenCV decode buffer error!";
              *code = tron_status_imdecode_error;
              *err = const_cast<char *>(Tron::get_status_message(*code));
              return;
            }

      }else{
        LOG(WARNING) << "RequestData body is empty!";
        *code = tron_status_request_data_body_empty;
        *err = const_cast<char *>(Tron::get_status_message(*code));
        return;
      }
   }
   // Check responses buffer size must be not larger than 4M bytes
   if (requests_.requests_size()==0){
      LOG(WARNING) << "RequestData data is empty!";
      *code = tron_status_request_data_body_empty;
      *err = const_cast<char *>(Tron::get_status_message(*code));
      return;
   }
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
