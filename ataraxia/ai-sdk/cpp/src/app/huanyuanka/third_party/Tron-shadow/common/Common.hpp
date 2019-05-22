#ifndef __COMMON_HPP__
#define __COMMON_HPP__

#include <string>
#include <string.h>


namespace Shadow
{
enum ShadowStatus
{
    //common status
    shadow_status_success = 200,
    shadow_status_set_gpu_error = 501,
    shadow_status_host_malloc_error = 502,
    shadow_status_cuda_malloc_error = 503,
    shadow_status_create_stream_error = 504,
    shadow_status_deserialize_error = 505,
    shadow_status_blobname_error = 506,
    shadow_status_batchsize_exceed_error = 507,
    shadow_status_batchsize_zero_error = 508,
    shadow_status_cuda_memcpy_error = 509,
    shadow_status_invalid_gie_file = 510,
    shadow_status_cuda_free_error = 511,
    //mixup new add
    shadow_status_not_implemented = 517,
    shadow_status_initpara_error = 518,
    shadow_status_layername_error = 519,
    shadow_status_results_size_error = 520,
    //landmark status
    shadow_status_data_error = 512,
    shadow_status_invalid_uff_file = 513,
    shadow_status_create_model_error = 514,
    shadow_status_binding_size_error = 515,
    shadow_status_parse_landmark_error = 516,
};

inline const char *get_status_message(int code)
{
    switch (code)
    {
    //common status
    case 200:
        return "shadow_status_success";
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
    //landmark status
    case 512:
        return "shadow_status_data_error";
    case 513:
        return "shadow_status_invalid_uff_file";
    case 514:
        return "shadow_status_create_model_error";
    case 515:
        return "shadow_status_binding_size_error";
    case 516:
        return "shadow_status_parse_landmark_error";
    //mixup new add
    case 517:
        return "shadow_status_not_implemented";
    case 518:
        return "shadow_status_initpara_error";
    case 519:
        return "shadow_status_layername_error";
    case 520:
        return "shadow_status_results_size_error";
    default:
        return "Unknown error";
    }
}

} // namespace Shadow
#endif /* Common_h */

