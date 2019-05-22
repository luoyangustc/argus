#ifndef TRON_FACE_SEARCH_INFER_HPP
#define TRON_FACE_SEARCH_INFER_HPP

#include <array>
#include <map>
#include <string>
#include <utility>
#include <vector>

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
namespace fs {

struct SetParams {
  SetParams() {}
  SetParams(std::string key, int size_limit, std::array<double, 3> threshold)
      : key(key), size_limit(size_limit), threshold(threshold) {}

  std::string key;
  int size_limit = 0;
  std::array<double, 3> threshold;
};

struct CustomParams {
  CustomParams() {}
  ~CustomParams() {}

  std::map<std::string, SetParams> sets;
  std::array<double, 3> threshold;
};

struct Feature {
  int index;
  std::string key;
  std::string label;
  std::string group;  // 后续应该移至label
  std::array<double, 512> feature;

  std::string sample_url;
  std::array<std::array<int, 2>, 4> sample_pts;
  std::string sample_id;
};

struct Handle {
  const int responses_max_buffer_size_ = 4 * 1024 * 1024;

  int batch_size;
  CustomParams *params;

  std::vector<std::pair<int, std::vector<Feature>>> *features;

  std::vector<char> out_data;
};

struct Context {
  int batch_size;
  CustomParams params;

  std::vector<std::pair<int, std::vector<Feature>>> features;

  std::vector<Handle *> handles;
};

}  // namespace fs
}  // namespace tron

#endif  // TRON_FACE_SEARCH_INFER_HPP NOLINT
