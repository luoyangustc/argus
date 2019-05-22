#ifndef TRON_FOO_INFERENCE_HPP
#define TRON_FOO_INFERENCE_HPP

#include <vector>

#include <opencv2/opencv.hpp>

#include "common/boxes.hpp"
#include "common/type.hpp"
#include "forward.hpp"
#include "framework/base.hpp"
#include "framework/forward.hpp"

namespace tron {
namespace foo {

class Inference
    : public tron::framework::InferenceBase<
          framework::ForwardWrap<Forward,
                                 inference::foo::ForwardRequest,
                                 inference::foo::ForwardResponse>,
          cv::Mat,
          float> {
  using Request = inference::foo::ForwardRequest;
  using Response = inference::foo::ForwardResponse;
  using ForwardWrap = framework::ForwardWrap<Forward, Request, Response>;
  using Base = tron::framework::InferenceBase<ForwardWrap, cv::Mat, float>;

 public:
  Inference() = default;
  ~Inference() {}

  void Predict(const std::vector<cv::Mat> &im_mats,
               std::vector<float> *outs) override;
};

}  // namespace foo
}  // namespace tron

#endif  // TRON_FOO_INFERENCE_HPP NOLINT
