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
#include "../../third_party/mix/include/Net.hpp"
#include <opencv2/opencv.hpp>
using namespace Shadow;

namespace Tron{

struct TronInferHandle {

  // Serving limit the maximum response buffer size to 4M bytes
  const int responses_max_buffer_size_ = 4 * 1024 * 1024;
  // Magic method for your task, change this to your algorithm implement
  Net * net_;
  //net num
  int model_num_ = 1;
  // gpu index
  int gpu_id_ = 0;
};

enum TronStatus {
  tron_status_success = 200,
  tron_status_method_nullptr = 400,
  tron_status_parse_model_error = 401,
  tron_status_request_data_body_empty = 402,
  tron_status_imdecode_error = 403,
  tron_status_image_size_error = 404,
  tron_status_parse_custom_error = 405,
  tron_status_response_buffer_not_enough = 500
};


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
        return "tron_status_exceed_max_batchsize_error";
  case 406:
        return "tron_status_parse_custom_error";
  case 500:
        return "tron_status_response_buffer_not_enough";
  case 501:
        return "shadow_status_set_gpu_error";
  case 502:
        return "shadow_status_host_malloc_error";
  case 503:
        return "shadow_status_cuda_malloc_error";
  case 504:
        return "shadow_status_create_tream_error";
  case 505:
        return "shadow_status_deserialize_error";
  case 506:
        return "shadow_status_blobname_error";
  case 507:
        return "shadow_status_batchsize_exceed_error";
  case 508:
        return "shadow_status_batchsize_zero_error";
  case 509:
        return "shadow_status_cuda_memcpy_error";
  case 510:
        return "shadow_status_invalid_gie_file";
  case 511:
        return "shadow_status_cuda_free_error";
  default:
        return "Unknown error";
    }
  }

inline int tron_error_round(int code) {
  return code>400&&code<500?400:500;  //400 tron error  and  500 shadow error
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

inline bool parse_create_params(void *params, int params_size,
                                std::string &custom_params,
                                vector<std::vector<char>> &engin,vector<int> &engin_size) {
  inference::CreateParams create_params;
  bool success = read_proto_from_array(params, params_size, &create_params);
  if (success) {
    custom_params = create_params.custom_params();
    if (create_params.model_files_size() > 0) {
      engin.resize(create_params.model_files_size());
      LOG(INFO)<<create_params.model_files_size()<<" ENGINS";
      engin_size.resize(create_params.model_files_size());
      for(int i = 0;i<create_params.model_files_size();i++){
        const auto &model_file = create_params.model_files(i);
        if (model_file.has_body()) {
          int j = i;
          if (model_file.name().find("mix_engin.bin") != std::string::npos) {
            j = 0;
          } else if (model_file.name().find("onet.bin") != std::string::npos) {
            j = 1;
          } else if (model_file.name().find("face-feature-res18.bin") != std::string::npos) {
            j = 2;
          }
          engin_size[j] = model_file.body().size();
          const auto *body_data = model_file.body().data();
          engin.at(j).resize(engin_size[j]);
          memcpy((void*)engin.at(j).data(),body_data,engin_size[j]);
          LOG(INFO) <<model_file.name() << " Model file's body size: " << engin_size[j];
          LOG(INFO) <<model_file.name() << " Model file's body md5: " << MD5(model_file.body());
        } else {
          LOG(WARNING) << model_file.name()<< " Model file has no body!";
          return false;
        }
      }
      return true;
    } else {
      LOG(WARNING) << "CreateParams has no model files!";
      return false;
    }
  } else {
    LOG(WARNING) << "Parsing CreateParams Error!";
    return false;
  }
}

inline bool parse_custom_params(const std::string custom_params,
                                TronInferHandle *handle) {
  if (!custom_params.empty()) {
    // Parse some useful params
    const auto &document = Tron::get_document(custom_params);
    if(document.HasMember("gpu_id")) {
        handle->gpu_id_ = Tron::get_int(document, "gpu_id", 0);
    }
    if(document.HasMember("model_num")) {
        handle->model_num_ = Tron::get_int(document, "model_num", 0);
    }
  }else
  {
    return true;
  }
}

}  // namespace Tron

#endif  // TRON_COMMON_HELPER_HPP
