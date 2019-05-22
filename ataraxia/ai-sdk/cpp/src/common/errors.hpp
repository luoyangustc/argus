#ifndef ERRORS_H_
#define ERRORS_H_

namespace tron {

enum TronStatus {
  tron_status_success = 200,

  tron_status_method_nullptr = 400,
  tron_status_parse_model_error = 401,
  tron_status_request_data_body_empty = 402,
  tron_status_imdecode_error = 403,
  tron_status_image_size_error = 404,
  tron_status_response_buffer_not_enough = 405,
  tron_status_refinenet_output_rect_error = 406
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
      return "tron_status_response_buffer_not_enough";
    case 406:
      return "tron_status_refinenet_output_rect_error";
    default:
      return "Unknown error";
  }
}
}  // namespace tron

#endif  // ERRORS_H__
