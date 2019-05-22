#ifndef TRON_TERROR_MIXUP_INFER_HPP
#define TRON_TERROR_MIXUP_INFER_HPP

#include <string>
#include <vector>
#include "framework/context.hpp"

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

namespace tron {
namespace terror_mixup {

using std::string;
using std::vector;

struct CustomParams {
  std::string frontend;
  std::string backend;
  int gpu_id = 0;
  vector<string> files;  // workaround
};

template <typename Archiver>
Archiver &operator&(Archiver &ar, CustomParams &p) {  // NOLINT
  ar.StartObject();
  if (ar.HasMember("frontend")) ar.Member("frontend") & p.frontend;
  if (ar.HasMember("backend")) ar.Member("backend") & p.backend;
  if (ar.HasMember("gpu_id")) ar.Member("gpu_id") & p.gpu_id;
  CHECK(ar) << "parse CustomParams";
  return ar.EndObject();
}

using Config = tron::framework::Config<CustomParams>;
using Context = tron::framework::Context<CustomParams>;
using Handle = tron::framework::Handle<CustomParams>;

}  // namespace terror_mixup
}  // namespace tron

#endif  // TRON_TERROR_MIXUP_INFER_HPP NOLINT
