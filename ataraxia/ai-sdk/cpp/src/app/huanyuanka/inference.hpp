#pragma once

#include <vector>

#include <opencv2/opencv.hpp>

#include "inference_ff.hpp"
#include "inference_fo.hpp"
#include "inference_mix.hpp"

namespace tron {
namespace mix {

class Inference {
 public:
  int Predict(const std::vector<cv::Mat> &requests,
              std::vector<ResponseMix> *responses);

  std::shared_ptr<InferenceMix> mix_;
  std::shared_ptr<InferenceFO> fo_;
  std::shared_ptr<InferenceFF> ff_;
};

}  // namespace mix
}  // namespace tron
