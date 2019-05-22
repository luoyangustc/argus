#ifndef TRON_FACE_FEATURE_INFER_HPP
#define TRON_FACE_FEATURE_INFER_HPP

#include <string>
#include <vector>
#include "inference.hpp"

#ifdef __cplusplus
extern "C" {
#endif

typedef void *PredictorContext;
typedef void *PredictorHandle;

const char *QTGetLastError();

int QTPredCreate(const void *, const int, PredictorContext *);

int QTPredHandle(PredictorContext, const void *, const int, PredictorHandle *);

int QTPredInference(PredictorHandle, const void *, const int, void **, int *);

int QTPredFree(PredictorContext);

#ifdef __cplusplus
}
#endif

#endif  // TRON_FACE_FEATURE_INFER_HPP NOLINT
