#ifndef TRON_WA_INFER_HPP  // NOLINT
#define TRON_WA_INFER_HPP

#include <map>
#include <string>
#include <utility>
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

#endif  // TRON_WA_INFER_HPP NOLINT
