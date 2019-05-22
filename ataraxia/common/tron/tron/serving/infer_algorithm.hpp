#ifndef TRON_EXPORT_INFER_ALGORITHM_HPP
#define TRON_EXPORT_INFER_ALGORITHM_HPP

#ifdef __cplusplus
extern "C" {
#endif

/**
 * void initEnv(InitParams params,
 *              const int params_size,
 *              int* code,
 *              char** err);
 */
void initEnv(void *params, const int params_size, int *code, char **err);

/**
 * void* createNet(CreateParams params,
 *                 const int params_size,
 *                 int* code,
 *                 char** err);
 */
void *createNet(void *params, const int params_size, int *code, char **err);

/**
 * void netPreprocess(const void* ctx,
 *                    InferenceRequest request,
 *                    const int request_size,
 *                    int* code,
 *                    char** err,
 *                    InferenceResponses ret,
 *                    int* ret_size);
 */
// void netPreprocess(const void *ctx, void *request, const int request_size,
//                   int *code, char **err, void *ret, int *ret_size);

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
                  int *code, char **err, void *ret, int *ret_size);

#ifdef __cplusplus
}
#endif

#endif  // TRON_EXPORT_INFER_ALGORITHM_HPP
