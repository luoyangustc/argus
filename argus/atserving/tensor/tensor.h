#ifndef __DETECT_H__
#define __DETECT_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>

// classify interface
typedef struct tensorNet_ctx classifier_ctx;

classifier_ctx* classifier_initialize(char* model_file, char* trained_file,
                                      char* mean_file, char* label_file, size_t batch_size);

const char* classifier_classify(classifier_ctx* ctx, char* image_file, size_t length);

// detect interface
typedef struct tensorNet_ctx detecter_ctx;

detecter_ctx* detecter_initialize(char* model_file, char* trained_file, char* label_file, size_t batch_size);

const char* detecter_detect(detecter_ctx* ctx, char* image_file, size_t length);

#ifdef __cplusplus
}
#endif
#endif