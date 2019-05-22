#ifndef TRON_FRAMEWORK_BASE_HPP  // NOLINT
#define TRON_FRAMEWORK_BASE_HPP

#include <vector>
#include "common/type.hpp"
#include "glog/logging.h"

namespace tron {
namespace framework {

class Void {
 public:
  Void() = default;
  ~Void() {}
};

template <class Request, class Response, class Config = Void>
class ForwardBase {
  typename std::vector<Request>::const_iterator RequestIterator;
  typename std::vector<Response>::const_iterator ResponseIterator;

 public:
  virtual ~ForwardBase() {}
  virtual void Setup(const std::vector<std::vector<char>> &net_param_data,
                     const VecInt &in_shape, const int &gpu_id,
                     const Config &config = Config()) = 0;
  void Predict(const std::vector<Request> &, std::vector<Response> *);
  void Init() { init_func_(); }
  virtual void Release() = 0;

 protected:
  void Setup(const std::vector<int> &in_shape,
             std::function<void()> init_func = []() {});
  virtual void Process(typename std::vector<Request>::const_iterator,
                       typename std::vector<Request>::const_iterator,
                       typename std::vector<Response>::iterator) = 0;

  int batch_, in_num_, in_c_, in_h_, in_w_;
  VecFloat in_data_;
  std::function<void()> init_func_;
};

template <class Forward, class Request, class Response, class Config = Void>
class InferenceBase {
 public:
  virtual ~InferenceBase() {}
  void Setup(const std::vector<int> &in_shape,
             std::shared_ptr<Forward> forward,
             const Config &config = Config());
  virtual void Predict(const std::vector<Request> &,
                       std::vector<Response> *) = 0;

 protected:
  std::shared_ptr<Forward> forward_;
  VecFloat in_data_;
  int batch_, in_num_, in_c_, in_h_, in_w_;
  Config config_;
};

template <class Request, class Response, class Config>
void ForwardBase<Request, Response, Config>::Setup(
    const std::vector<int> &in_shape,
    std::function<void()> init_func) {
  CHECK_EQ(in_shape.size(), 4);
  batch_ = in_shape[0];
  in_c_ = in_shape[1], in_h_ = in_shape[2], in_w_ = in_shape[3];
  in_num_ = in_c_ * in_h_ * in_w_;
  in_data_.resize(batch_ * in_num_);
  init_func_ = init_func;
}

template <class Request, class Response, class Config>
void ForwardBase<Request, Response, Config>::Predict(
    const std::vector<Request> &requests,
    std::vector<Response> *responses) {
  for (std::size_t i = 0; i < requests.size(); i += batch_) {
    if (i + batch_ > requests.size()) {
      Process(requests.begin() + i, requests.end(), responses->begin() + i);
    } else {
      Process(requests.begin() + i, requests.begin() + i + batch_,
              responses->begin() + i);
    }
  }
}

template <class Forward, class Request, class Response, class Config>
void InferenceBase<Forward, Request, Response, Config>::Setup(
    const std::vector<int> &in_shape,
    std::shared_ptr<Forward> forward,
    const Config &config) {
  CHECK_EQ(in_shape.size(), 4);
  batch_ = in_shape[0];
  in_c_ = in_shape[1], in_h_ = in_shape[2], in_w_ = in_shape[3];
  in_num_ = in_c_ * in_h_ * in_w_;
  in_data_.resize(batch_ * in_num_);

  forward_ = forward;
  config_ = config;
}

}  // namespace framework
}  // namespace tron

#endif  // TRON_FRAMEWORK_BASE_HPP NOLINT