#ifndef __INFERENCE_H__
#define __INFERENCE_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>

// void initEnv(
//     InitParams params, const int params_size,
//     int* code, char** err);
void initEnv(
	void* params, const int params_size,
	int* code, char** err);

// void* createNet(
//     CreateParams params, const int params_size,
//     int* code, char** err);
void* createNet(
	void* params, const int params_size,
	int* code, char** err);

// InferenceRequest netPreprocess(
//     const void* ctx, InferenceRequest request, const int request_size,
//     int* code, char** err, void* ret, int* ret_size);
void netPreprocess(
	const void* ctx, void* request, const int request_size,
	int* code, char** err, void* ret, int* ret_size);

// InferenceResponses netInference(
// 	const void* net, InferenceRequests requests, const int requests_size,
// 	int* code, char** err, void* ret, int* ret_size);
void netInference(
	const void* net, void* requests, const int requests_size,
	int* code, char** err, void* ret, int* ret_size);

#ifdef __cplusplus
}
#endif

#endif
