#include <cstdio>
#include <cstddef>
#include "proto/inference.pb.h"
#include "vss.hpp"

void *fakeCtx;

void *VSSCreate(void *params, const int params_size, int *code, char **err) {
    printf("vss Create\n");
    fakeCtx = new(int);
    *code = vss_status_success;
    return fakeCtx;
}

void VSSAddStream(const void *ctx, void *request, const int request_size,
                   int *code, char **err) {
    printf("vss AddStream\n");
    *code = vss_status_success;
    return;
}

void VSSStopStream(const void *ctx, void *request, const int request_size,
               int *code, char **err) {
    printf("vss StopStream\n");
    *code = vss_status_success;
    return;
}

void VSSGetStatus(const void *ctx, void *ret, int *ret_size, int *code, char **err) {
    printf("vss GetStatus\n");
    inference::InferenceResponse inference_response;
    inference_response.set_result(R"(
        {
            "test_channel_1": 1,
            "test_channel_2": 1
        }
    )");

    int tmp_ret_size = 0;
    inference_response.SerializeToArray(ret, inference_response.ByteSize());
    *ret_size = inference_response.ByteSize();
    *code = vss_status_success;

    printf("VSSGetStatus ret_size: %d\n", *ret_size);
    return;
}

void VSSProcess(const void *ctx, void *ret, int *ret_size, int *code, char **err) {
    printf("vss Process\n");
    inference::InferenceResponse inference_response;
    auto buf = inference_response.SerializeAsString();
    int tmp_ret_size = 0;
    inference_response.SerializePartialToArray(ret, tmp_ret_size);

    *code = vss_status_success;
    return;
}

void VSSRelease(const void *ctx) {
    printf("vss Release\n");
    auto *ptr = reinterpret_cast<void*>(const_cast<void *>(ctx));
    if ( ! ptr ) {
        delete ptr;
    }
    return;
}