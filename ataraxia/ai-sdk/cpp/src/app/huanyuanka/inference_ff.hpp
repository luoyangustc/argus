#pragma once

#include <opencv2/opencv.hpp>

#include "common/type.hpp"
#include "tensord/tensord.hpp"

namespace tron {
namespace mix {

class InferenceFF {
 public:
  InferenceFF() = default;
  ~InferenceFF() {}

  void Setup(std::shared_ptr<tensord::core::Engine<float>> engine,
             const std::vector<int> &in_shape);
  void Predict(const std::vector<cv::Mat> &requests,
               std::vector<std::vector<float>> *responses);

 private:
  std::shared_ptr<tensord::core::Engine<float>> engine_;
  VecFloat in_data_;
  int batch_, in_num_, in_c_, in_h_, in_w_;
};

}  // namespace mix
}  // namespace tron
