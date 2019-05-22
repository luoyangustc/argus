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
    Tron::ShadowDetectionRefineDet *fd_method_ = nullptr;
    std::vector<std::string> labels_;
    
    //mtcnn
    Tron::QualityEvalutation* qa_method_ = nullptr;
    //configuration
    bool const_use_quality=true;
    float neg_threshold=0;
    float pose_threshold=0;
    float cover_threshold=0;
    float blur_threshold=0.98;
    float quality_threshold=0.6;
    bool output_quality_score=false;
    int min_face=50;
    
    int batch_size=MAX_BATCH_SIZE;
    
};

int TronProcess(const cv::Mat &im_mat, TronInferHandle *handle,
                TronDetectionOutput *output, bool use_quality) {
    if (handle->fd_method_ == nullptr) {
        LOG(WARNING) << "Tron is uninitialized";
        return tron_status_method_nullptr;
    }
    
    int im_h = im_mat.rows, im_w = im_mat.cols;
    
    if (im_h <= 1 || im_w <= 1) {
        return tron_status_image_size_error;
    }
    // detect out
    std::vector<Tron::VecBoxF> Gboxes;
    Tron::VecRectF rois{Tron::RectF(0, 0, im_w, im_h)};
    handle->fd_method_->Predict(im_mat, rois, &Gboxes);
    output->objects.clear();
    // quality
    Tron::VecBoxF boxes;
    std::vector<int> Qalabels;
    std::vector<std::vector<float>> Qaprob;
    std::vector<int> Orientation;
    if(use_quality&&Gboxes[0].size()>0){
        if (handle->qa_method_ == nullptr){
            LOG(WARNING) << "Tron quality is uninitialized";
            return tron_status_method_nullptr;
        }
        handle->qa_method_->Predict(im_mat,Gboxes[0],Qaprob,Qalabels,Orientation,handle->min_face);
    }
    
    boxes=Gboxes[0];
    
    for (int iter=0;iter<boxes.size();iter++){
        if(!use_quality){//only facedetection
            const auto box = boxes[iter];
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
        }else{
            if(Qalabels[iter]!=1){//not neg face
                const auto box = boxes[iter];
                TronRectangle rect = {};
                rect.xmin = static_cast<int>(box.xmin);
                rect.xmax = static_cast<int>(box.xmax);
                rect.ymin = static_cast<int>(box.ymin);
                rect.ymax = static_cast<int>(box.ymax);
                rect.id = box.label;
                rect.score = box.score;
                rect.quality_category = -1;
                if (box.label >= 0 && box.label < handle->labels_.size()) {
                    rect.label = handle->labels_[box.label];
                } else {
                    rect.label = "";
                }
                if(!handle->output_quality_score){
                    rect.quality_category=Qalabels[iter];
                    for(int i=0;i<5;++i){
                        rect.quality_cls[i]=-1;
                    }
                }else{
                    const auto& quality_prob=Qaprob[iter];
                    rect.quality_category=Qalabels[iter];
                    for(int i=0;i<5;++i){
                        rect.quality_cls[i]=quality_prob[i];
                    }
                }
                rect.orient_category=Orientation[iter];
                output->objects.push_back(rect);
            }//if(Qalabels[iter]!=1)
        }
    }
    return tron_status_success;
}

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

void initEnv(void *params, const int params_size, int *code, char **err) {}

/**
 * void* createNet(CreateParams params,
 *                 const int params_size,
 *                 int* code,
 *                 char** err);
 */
void *createNet(void *params, const int params_size, int *code, char **err) {
    LOG(INFO) << "Starting create fdnet!";
    
    // Check some pointers not be nullptr
    CHECK_NOTNULL(code) << "code is nullptr";
    CHECK_NOTNULL(err) << "err is nullptr";
    
    // Initial a handle
    auto *handle_ = new TronInferHandle();
    
    // Parse CreateParams data
    std::string custom_params;
    tron::MetaNetParam meta_net_param_fd,meta_net_param_qa;
    bool success = Tron::parse_create_params(params, params_size, &custom_params,
                                             &meta_net_param_fd,&meta_net_param_qa);
    
    if (!success) {
        *err = const_cast<char *>(Tron::get_status_message(tron_status_parse_model_error));
        *code = tron_status_parse_model_error>400&&tron_status_parse_model_error<500?400:500;
        return nullptr;
    }
    
    // Parse some useful params
    const auto &document = Tron::get_document(custom_params);
    if(document.HasMember("gpu_id")) {
        handle_->gpu_id_ = Tron::get_int(document, "gpu_id", 0);
    }
    if(document.HasMember("const_use_quality")) {
        handle_->const_use_quality = Tron::get_int(document, "const_use_quality", 1);
    }
    if(document.HasMember("blur_threshold")) {
        handle_->blur_threshold = Tron::get_float(document, "blur_threshold",0.98);
    }
    if(document.HasMember("output_quality_score")) {
        handle_->output_quality_score = Tron::get_int(document, "output_quality_score", 1);
    }
    if(document.HasMember("min_face")) {
        handle_->min_face = Tron::get_int(document, "min_face", 50);
    }
    
    LOG(INFO)<<"gpu: "<<handle_->gpu_id_
    <<",use_quality= "<<handle_->const_use_quality
    <<",blur_threshold="<< handle_->blur_threshold
    <<",min_face="<< handle_->min_face
    <<",output_quality_score="<<handle_->output_quality_score;
    
    handle_->fd_method_ = new Tron::ShadowDetectionRefineDet();
    handle_->fd_method_->Setup(meta_net_param_fd, {1}, handle_->gpu_id_);
    
    if(1){
        LOG(INFO) << "Starting create qualitynet!";
        handle_->qa_method_ = new Tron::QualityEvalutation();
        handle_->qa_method_->Setup(meta_net_param_qa, {handle_->batch_size}, handle_->gpu_id_,
                                   handle_->neg_threshold,
                                   handle_->pose_threshold,
                                   handle_->cover_threshold,
                                   handle_->blur_threshold,
                                   handle_->quality_threshold);
    }
    
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
                bool use_quality = ptr->const_use_quality;
                if(requests_.requests(n).has_params()){
                    const std::string request_params=requests_.requests(n).params();
                    const auto document = Tron::get_document(request_params);
                    if (document.HasMember("use_quality"))
                    {
                        use_quality = document["use_quality"].GetInt();
                    }
                }

                TronDetectionOutput detection_output = {};
                int status = TronProcess(im_mat, ptr, &detection_output, use_quality);
                if (status == tron_status_success) {
                    // Transform result struct to json string
                    const auto &result_json_str = Tron::get_detect_json(use_quality,ptr->output_quality_score,detection_output);
                    res->set_code(status);
                    res->set_message(Tron::get_status_message(status));
                    res->set_result(result_json_str);
                } else {
                    *err = const_cast<char *>(Tron::get_status_message(status));
                    *code = status>400&&status<500 ? 400 : 500;
                    return;
                }
            } else{
                LOG(WARNING) << "OpenCV decode buffer error!";
                *err = const_cast<char *>(Tron::get_status_message(tron_status_imdecode_error));
                *code = tron_status_imdecode_error>400&&tron_status_imdecode_error<500?400:500;
                return;
            }
        } else {
            LOG(WARNING) << "RequestData body is empty!";
            *err = const_cast<char *>(Tron::get_status_message(tron_status_request_data_body_empty));
            *code = tron_status_request_data_body_empty>400&&tron_status_request_data_body_empty<500?400:500;
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
        *err = const_cast<char *>(Tron::get_status_message(tron_status_response_buffer_not_enough));
        *code = tron_status_response_buffer_not_enough>400&&tron_status_response_buffer_not_enough<500?400:500;
    }
}
