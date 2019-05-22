#ifndef VSS_VSS_HPP
#define VSS_VSS_HPP


#ifdef __cplusplus
extern "C" {
#endif

const int responses_max_buffer_size = 1024 * 1024 * 1024;

enum VSSStatus {
    vss_status_success = 200,
    vss_status_method_nullptr = 550,
    vss_status_parse_model_error = 551,
    vss_status_request_data_uri_empty = 552,
    vss_status_request_data_attribute_empty = 553,
    vss_status_request_data_body_empty = 554,
    vss_status_imdecode_error = 555,
    vss_status_image_size_error = 556,
    vss_status_open_file_error = 557,
    vss_status_response_buffer_not_enough = 558,
    vss_status_infer_timeout = 559,
    vss_status_stream_stop = 560,
    vss_status_push_stream_error = 561,
};


/**
 * void* VSSCreate(CreateParams params,
 *                 const int params_size,
 *                 int* code,
 *                 char** err);
 */
void *VSSCreate(void *params, const int params_size, int *code, char **err);

/**
 * void AddStream(void* ctx,
 *                   InferenceRequest request,
 *                   const int request_size,
 *                   int* code,
 *                   char** err);
 */
void VSSAddStream(const void *ctx, void *request, const int request_size,
                  int *code, char **err);

void VSSStopStream(const void *ctx, void *request, const int request_size,
                   int *code, char **err);

void VSSGetStatus(const void *ctx, void *ret, int *ret_size, int *code, char **err);
/**
 * void VSSProcess(void* ctx,
 *                      InferenceResponse ret,
 *                      int* ret_size,
 *                      int* code,
 *                      char** err);
 */
void VSSProcess(const void *ctx, void *ret, int *ret_size, int *code, char **err);

void VSSRelease(const void *ctx);


#ifdef __cplusplus
}
#endif

#endif //VSS_VSS_HPP