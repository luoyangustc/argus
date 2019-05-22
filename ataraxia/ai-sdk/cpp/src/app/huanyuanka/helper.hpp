#pragma once

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <opencv2/opencv.hpp>

#include "Net.hpp"
#include "proto/inference.pb.h"

namespace tron {
namespace mix {

struct TronInferHandle {
  // Serving limit the maximum response buffer size to 4M bytes
  const int responses_max_buffer_size_ = 4 * 1024 * 1024;
  // Magic method for your task, change this to your algorithm implement
  Shadow::Net *net_;
  // net num
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
  // 400 tron error  and  500 shadow error
  return code > 400 && code < 500 ? 400 : 500;
}

}  // namespace mix
}  // namespace tron
