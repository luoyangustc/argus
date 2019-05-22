#pragma once

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
