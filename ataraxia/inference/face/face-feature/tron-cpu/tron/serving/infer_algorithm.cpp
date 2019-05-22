#include "infer_algorithm.hpp"
#include "face_alignment.hpp"

#include "infer_implement.hpp"

#include "common/helper.hpp"

//#define DEBUG_LOG

using namespace Tron;

struct TronInferHandle {
  // Serving limit the maximum response buffer size to 4M bytes
  const int responses_max_buffer_size_ = 4 * 1024 * 1024;
  // gpu index
  int gpu_id_ = 0;
  // Magic method for your task, change this to your algorithm implement
  Tron::MTCNN      *mtcnn_ = nullptr;
  Tron::Feature  *feature_ = nullptr;
};

// MTCNN Function used to do the inference
int TronProcessMTCNN(const cv::Mat &im_mat, TronInferHandle *handle,
                     const VecBoxF &face_boxes,
                     std::vector<std::vector<cv::Point2d>> *cv_faces_points) {
  if (handle->mtcnn_ == nullptr) {
    LOG(WARNING) << "Tron mtcnn is uninitialized";
    return tron_status_method_nullptr;
  }

  int im_h = im_mat.rows, im_w = im_mat.cols;

  if (im_h <= 1 || im_w <= 1) {
    return tron_status_image_size_error;
  }

  std::vector<VecPointF> tron_faces_points;
  handle->mtcnn_->Predict(im_mat, face_boxes, &tron_faces_points);

  cv_faces_points->clear();
  for (const auto &tron_face_points : tron_faces_points) {
    std::vector<cv::Point2d> cv_face_points;
    for (const auto &point : tron_face_points) {
      cv_face_points.emplace_back(point.x, point.y);
    }
    cv_faces_points->push_back(cv_face_points);
  }

  return tron_status_success;
}

// Feature Function used to do the inference
int TronProcessFeature(const cv::Mat &im_mat, TronInferHandle *handle,
                       std::vector<float> *features) {
  if (handle->feature_ == nullptr) {
    LOG(WARNING) << "Tron feature is uninitialized";
    return tron_status_method_nullptr;
  }

  int im_h = im_mat.rows, im_w = im_mat.cols;

  if (im_h <= 1 || im_w <= 1) {
    return tron_status_image_size_error;
  }

  std::vector<std::map<std::string, Tron::VecFloat>> Gscores;
  Tron::VecRectF rois{Tron::RectF(0, 0, im_w, im_h)};
  handle->feature_->Predict(im_mat, rois, &Gscores);

  *features = Gscores[0].at("score");

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
  LOG(INFO) << "facex-feature-tron-cpu: createNet() start";

  // Check some pointers not be nullptr
  CHECK_NOTNULL(code) << "code is nullptr";
  CHECK_NOTNULL(err) << "err is nullptr";

  // Initial a handle
  auto *handle_ = new TronInferHandle();

  // Parse CreateParams data
  std::string custom_params;
  tron::MetaNetParam meta_net_param_mtcnn, meta_net_param_feature;

  bool success =
      Tron::parse_create_params(params, params_size, &custom_params,
                                &meta_net_param_mtcnn, &meta_net_param_feature);

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

  // Initial network
  handle_->mtcnn_ = new Tron::MTCNN();
  handle_->mtcnn_->Setup(meta_net_param_mtcnn, {1}, handle_->gpu_id_);
  //LOG(INFO)<<"after MTCNN::Setup()\n";
  handle_->feature_ = new Tron::Feature();
  handle_->feature_->Setup(meta_net_param_feature, {1}, handle_->gpu_id_);

  *code = tron_status_success;
  *err = const_cast<char *>(Tron::get_status_message(*code));

  LOG(INFO) << "facex-feature-tron-cpu: createNet() finished";

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
  LOG(INFO) << "facex-feature-tron-cpu: netInference() start";

  // Check some pointers not be nullptr
  CHECK_NOTNULL(code) << "code is nullptr";
  CHECK_NOTNULL(err) << "err is nullptr";
  CHECK_NOTNULL(ret) << "ret is nullptr";
  CHECK_NOTNULL(ret_size) << "ret_size is nullptr";

  // Parse InferenceRequests data
  auto *ptr = reinterpret_cast<TronInferHandle *>(const_cast<void *>(ctx));
  inference::InferenceRequests requests_;
  requests_.ParseFromArray(requests, requests_size);
//  std::vector<std::vector<float> > face_features_output; // face features of many persons
  inference::InferenceResponses responses_;
  for (int n = 0; n < requests_.requests_size(); ++n) {
    const auto &req_data = requests_.requests(n).data();
    auto *res = responses_.add_responses();

    if (req_data.has_uri()) {
      LOG(INFO) << "facex-feature-tron-cpu: request image uri: " << req_data.uri();
    }
    if (req_data.has_attribute()) {
      LOG(INFO) << "facex-feature-tron-cpu: request attribute: " << req_data.attribute();
    }
    
    if (req_data.has_body() && !req_data.body().empty()) {
      const auto &im_mat = decode_image_buffer(req_data.body());
      if (!im_mat.empty()) {
        // Parse request face boxes from attribute
        
        VecBoxF face_boxes,face_boxes_;

        bool success = Tron::parse_request_boxes(req_data.attribute(), &face_boxes_);
        if (!success) {
          *code = tron_status_request_data_attribute_error;
          *err = const_cast<char *>(Tron::get_status_message(*code));
          return;
        }

        // scale the refinedet rect to o-net input.
        // const static float scale_x_min=0.8,scale_x_max=1.1,
        //                    scale_y_min=0.8;
        const int image_w=im_mat.cols,image_h=im_mat.rows;
        auto&  box_info=face_boxes_[0];
//        for( auto& box_info:face_boxes_) {

        const int w=box_info.xmax-box_info.xmin;
        const int h=box_info.ymax-box_info.ymin;
        const int diff=abs(w-h);

        // make sure the box's 4 pts are all inside the image
        if(box_info.xmin<0 || box_info.xmax>image_w
          || box_info.ymin<0 || box_info.ymax>image_h
          || w>image_w || h>image_h)
        {
          LOG(WARNING) << "All of face box's pts must be inside the image's region! ";
          *code = tron_status_request_data_attribute_error;
          *err = const_cast<char *>(Tron::get_status_message(*code));
          return;
        }

        if (image_h < 20 || image_w < 20) {
          LOG(WARNING) << "Image width and height must be >=20 !";
          *code = tron_status_image_size_error;
          *err = const_cast<char *>(Tron::get_status_message(*code));
          return;
        }

        //LOG(INFO)<<"w="<<w<<",h="<<h<<",diff="<<diff<<"\n";
        if(w>h){
          box_info.ymin-=diff/2;
          box_info.ymax+=diff/2;

          if(box_info.ymin<0){
            box_info.ymin=0;
          }

          if (box_info.ymax>image_h-1) {
            box_info.ymax=image_h-1;
          }
        }
        else {
          box_info.xmin-=diff/2;
          box_info.xmax+=diff/2;
          
          if(box_info.xmin<0){
            box_info.xmin=0;
          }

          if (box_info.xmax>image_w-1) {
            box_info.xmax=image_w-1;
          }
        }
        face_boxes.push_back(box_info);

 //       }                  

#ifdef DEBUG_LOG
        LOG(INFO) << "---> run MTCNN to get 5 pts: " << endl;
#endif
        // Process landmark points
        std::vector<std::vector<cv::Point2d>> cv_faces_points;
        int status =TronProcessMTCNN(im_mat, ptr, face_boxes, &cv_faces_points);
        if (status != tron_status_success) {
          *code = status;
          *err = const_cast<char *>(Tron::get_status_message(*code));
          return;
        }

#ifdef DEBUG_LOG
        LOG(INFO) << "<--- MTCNN finished" << endl;
#endif

#ifdef DEBUG_LOG
        // print face landmarks
        int i=0;
        for (const auto &cv_face_points : cv_faces_points) {
          LOG(INFO) << "landmarks for face " << i << endl;
          i++;

          for (const auto &point : cv_face_points) {
            LOG(INFO) << "x=" << point.x << ", y=" << point.y << endl;
          }
        }
#endif

#ifdef DEBUG_LOG
        LOG(INFO) << "---> aligne faces by 5 pts: " << endl;
#endif
        // Process face alignment
        std::vector<cv::Mat> aligned_faces;
        faceAlignmet(im_mat, cv_faces_points, aligned_faces, FACE_112_112);


#ifdef DEBUG_LOG
        LOG(INFO) << "<--- face alignment finished" << endl;
#endif

#ifdef DEBUG_LOG
        LOG(INFO) << "---> extract face feature: " << endl;
#endif
        // Process face feature extraction
        std::vector<float> face_features;
        auto &aligned_face=aligned_faces[0];
    //    for (const auto &aligned_face:aligned_faces) {
        std::vector<float> features;
        status = TronProcessFeature(aligned_face, ptr, &features);
        if (status == tron_status_success) {
            // face_features.insert(face_features.end(), features.begin(),
            //                       features.end());
            //  for(int i=0;i<512;++i)
            //  LOG(INFO)<<features[i]<<" ";
            //  LOG(INFO)<<"\n\n\n";

            //  for(int i=0;i<512;++i)
            //   std::cout<<features[i]<<" ";
            //   std::cout<<"\n\n\n";

#ifdef DEBUG_LOG
            LOG(INFO) << "<--- feature extraction finished" << endl;
#endif

            const void* features_bytes = reinterpret_cast<const void*>(&features[0]);
            int features_bytes_size = (features.size())*4;
            res->set_body(features_bytes, features_bytes_size);

            // const void* features_bytes = reinterpret_cast<const void*>(&features[0]);
            // int size_classification = (features.size())*4;
            // res->set_body(features_bytes,size_classification);

            // Set results to response message
            res->set_code(tron_status_success);
            res->set_message(Tron::get_status_message(tron_status_success));

            //    res->set_body(result_json_str);

            //    res->set_body(face_features.data(),
            //                  face_features.size() * sizeof(float));

          } else {
            *code = status;
            *err = const_cast<char *>(Tron::get_status_message(*code));
            return;
          }
    // } // for

    //    face_features_output.push_back(face_features);
 
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
  }   // for 

  
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

  LOG(INFO) << "facex-feature-tron-cpu: netInference() finished";
}
