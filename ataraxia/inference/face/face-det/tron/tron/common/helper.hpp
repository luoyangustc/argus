#ifndef TRON_COMMON_HELPER_HPP
#define TRON_COMMON_HELPER_HPP

#include "common/boxes.hpp"
#include "common/json.hpp"
#include "common/log.hpp"
#include "common/md5.hpp"
#include "common/type.hpp"
#include "common/util.hpp"

#include "proto/inference.pb.h"

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

enum TronStatus {
    tron_status_success = 200,
    
    tron_status_method_nullptr = 400,
    tron_status_parse_model_error = 401 ,
    tron_status_request_data_body_empty = 402,
    tron_status_imdecode_error = 403,
    tron_status_image_size_error = 404,
    tron_status_response_buffer_not_enough = 405,
    tron_status_refinenet_output_rect_error= 406
};

struct TronRectangle {
    /** Rectangle location and label index. */
    int xmin, ymin, xmax, ymax, id;
    /** Rectangle score. */
    float score;
    /** Rectangle label. */
    std::string label;
    /* landmarks */
    float pts5[10];
    
    int category;
};

struct TronClassificationOutput {
    /** All class scores. */
    std::vector<float> scores;
    /** All class labels. */
    std::vector<std::string> labels;
};

struct TronDetectionOutput {
    /** All detected objects. */
    std::vector<TronRectangle> objects;
};

namespace Tron {
    
    inline const char *get_status_message(int code) {
        switch (code) {
            case 200:
                return "tron_status_success";
            case 400:
                return "tron_status_method_nullptr";
            case 401:
                return "tron_status_parse_model_error";
            case 402:
                return "tron_status_request_data_body_empty";
            case 403:
                return "tron_status_imdecode_error";
            case 404:
                return "tron_status_image_size_error";
            case 405:
                return "tron_status_response_buffer_not_enough";
            case 406:
                return "tron_status_refinenet_output_rect_error";
            default:
                return "Unknown error";
        }
    }
    
    /**
     * This function is used to transform classify struct results to json format
     */
    inline std::string get_classify_json(
                                         const TronClassificationOutput &classification_output, int top_k,
                                         float threshold) {
        const auto &scores = classification_output.scores;
        auto top_k_select = static_cast<int>(scores.size());
        if (top_k > 0) {
            top_k_select = top_k;
        }
        const auto &top_index = Util::top_k(scores, top_k_select);
        
        using namespace rapidjson;
        Document document;
        auto &alloc = document.GetAllocator();
        Value j_confidences(kObjectType), j_classes(kArrayType);
        for (const auto index : top_index) {
            auto score = scores[index];
            if (score < threshold) continue;
            Value j_class(kObjectType);
            j_class.AddMember("index", Value(index), alloc);
            j_class.AddMember("score", Value(score), alloc);
            if (!classification_output.labels.empty()) {
                const auto &label = classification_output.labels[index];
                if (!label.empty()) {
                    j_class.AddMember("class", Value(StringRef(label.c_str())), alloc);
                }
            }
            j_classes.PushBack(j_class, alloc);
        }
        j_confidences.AddMember("confidences", j_classes, alloc);
        StringBuffer buffer;
        Writer<StringBuffer> writer(buffer);
        j_confidences.Accept(writer);
        return std::string(buffer.GetString());
    }
    
    /**
     * This function is used to transform detect struct results to json format
     */
    inline std::string get_detect_json(bool output_quality,bool output_pts5,
                                       const TronDetectionOutput &detection_output){
        using namespace rapidjson;
        Document document;
        auto &alloc = document.GetAllocator();
        Value j_detections(kObjectType), j_rects(kArrayType);
        for (const auto &rect : detection_output.objects) {
            Value j_rect(kObjectType);
            j_rect.AddMember("index", Value(rect.id), alloc);
            j_rect.AddMember("score", Value(rect.score), alloc);
            j_rect.AddMember("class", Value("face"), alloc);
            
            Value lt(kArrayType), rt(kArrayType), rb(kArrayType), lb(kArrayType),pts(kArrayType);
            lt.PushBack(Value(rect.xmin), alloc).PushBack(Value(rect.ymin), alloc);
            rt.PushBack(Value(rect.xmax), alloc).PushBack(Value(rect.ymin), alloc);
            rb.PushBack(Value(rect.xmax), alloc).PushBack(Value(rect.ymax), alloc);
            lb.PushBack(Value(rect.xmin), alloc).PushBack(Value(rect.ymax), alloc);
            pts.PushBack(lt,alloc).PushBack(rt,alloc).PushBack(rb,alloc).PushBack(lb,alloc);
            j_rect.AddMember("pts", pts, alloc);
            
            if(output_quality){
                if(rect.category==0)
                j_rect.AddMember("quality", Value("clear"), alloc);
                if(rect.category==2)
                j_rect.AddMember("quality", Value("blur"), alloc);
                if(rect.category==3)
                j_rect.AddMember("quality", Value("pose"), alloc);
                if(rect.category==4)
                j_rect.AddMember("quality", Value("cover"), alloc);
                if(rect.category==5)
                j_rect.AddMember("quality", Value("small"), alloc);
            }
            if(output_pts5&&rect.category==0){
                Value pts5(kArrayType),left_eye(kArrayType),right_eye(kArrayType),
                nose(kArrayType),left_mouth(kArrayType),right_mouth(kArrayType);
                
                left_eye.PushBack(Value(rect.pts5[0]), alloc).PushBack(Value(rect.pts5[1]), alloc);
                right_eye.PushBack(Value(rect.pts5[2]), alloc).PushBack(Value(rect.pts5[3]), alloc);
                nose.PushBack(Value(rect.pts5[4]), alloc).PushBack(Value(rect.pts5[5]), alloc);
                left_mouth.PushBack(Value(rect.pts5[6]), alloc).PushBack(Value(rect.pts5[7]), alloc);
                right_mouth.PushBack(Value(rect.pts5[8]), alloc).PushBack(Value(rect.pts5[9]), alloc);
                pts5.PushBack(left_eye, alloc).PushBack(right_eye, alloc).PushBack(nose, alloc).PushBack(
                                                                                                         left_mouth, alloc).PushBack(right_mouth, alloc);
                
                j_rect.AddMember("pts5", pts5, alloc);
                
            }
            j_rects.PushBack(j_rect, alloc);
        }
        j_detections.AddMember("detections", j_rects, alloc);
        StringBuffer buffer;
        Writer<StringBuffer> writer(buffer);
        j_detections.Accept(writer);
        return std::string(buffer.GetString());
    }
    
    /**
     * This function is used to parse protobuf message from raw array
     */
    inline bool read_proto_from_array(const void *proto_data, int proto_size,
                                      google::protobuf::Message *proto) {
        using google::protobuf::io::ArrayInputStream;
        using google::protobuf::io::CodedInputStream;
        auto *param_array_input = new ArrayInputStream(proto_data, proto_size);
        auto *param_coded_input = new CodedInputStream(param_array_input);
        param_coded_input->SetTotalBytesLimit(INT_MAX, 1073741824);
        bool success = proto->ParseFromCodedStream(param_coded_input) &&
        param_coded_input->ConsumedEntireMessage();
        delete param_coded_input;
        delete param_array_input;
        return success;
    }
    
    /**
     * This function is used to parse the serving create params, parse the model
     * buffers and save them to /tmp/models
     */
    inline bool parse_create_params(void *params, int params_size,
                                    std::string *custom_params,
                                    tron::MetaNetParam* meta_net_param_fd,
                                    tron::MetaNetParam* meta_net_param_qa){
        inference::CreateParams create_params;
        bool success = read_proto_from_array(params, params_size, &create_params);
        if (success) {
            *custom_params = create_params.custom_params();
            LOG(INFO) << *custom_params;
            if (create_params.model_files_size() ==2) {
                
                int j=-1,qa_flag=-1,fd_flag=-1;
                for (int i=0; i<2; i++){
                    if (create_params.model_files(i).has_name()
                        && create_params.model_files(i).name().find("quality") != std::string::npos) {
                        qa_flag= i;
                        j=i;
                    }
                    else if (create_params.model_files(i).has_name()
                             && create_params.model_files(i).name().find("refinedet") != std::string::npos) {
                        fd_flag= i;
                    }
                }
                if(j<0){
                    LOG(FATAL) << "Cannot find a model file containing string 'quality' !";
                    return false;
                }
                
                const auto &model_file_fd = create_params.model_files(fd_flag);
                const auto &model_file_qa = create_params.model_files(qa_flag);
                bool parse_success=false;
                if(model_file_qa.has_body()) {
                    const auto *net_data = model_file_qa.body().data();
                    auto net_size = model_file_qa.body().size();
                    LOG(INFO) << "Quality Model file's body size: " << net_size;
                    LOG(INFO) << "Quality Model file's body md5: " << MD5(model_file_qa.body());
                    parse_success =
                    read_proto_from_array(net_data, net_size, meta_net_param_qa);
                } else {
                    LOG(WARNING) << "quality model file has no body!";
                    return false;
                }
                
                if (model_file_fd.has_body()) {
                    const auto *net_data = model_file_fd.body().data();
                    auto net_size = model_file_fd.body().size();
                    LOG(INFO) << "Facedetection Model file's body size: " << net_size;
                    LOG(INFO) << "Facedetection Model file's body md5: " << MD5(model_file_fd.body());
                    return read_proto_from_array(net_data, net_size, meta_net_param_fd);
                } else {
                    LOG(WARNING) << "Facedetection Model file has no body!";
                    return false;
                }
            } else {
                LOG(WARNING) << "CreateParams has no two model files!";
                return false;
            }
        } else {
            LOG(WARNING) << "Parsing CreateParams Error!";
            return false;
        }
    }
    
}  // namespace Tron

#endif  // TRON_COMMON_HELPER_HPP