#pragma once

#include <opencv2/opencv.hpp>

#include "common/type.hpp"
#include "tensord/tensord.hpp"

namespace tron {
namespace mix {

struct ResponseMix {
  int normal = 0;
  int march = 0;
  int text = 0;
  int face = 0;
  int bk = 0;
  int pulp = 0;
  VecBoxF boxes;
  std::vector<std::vector<float>> features;
};

class InferenceMix {
 public:
  InferenceMix() = default;
  ~InferenceMix() {}

  void Setup(std::shared_ptr<tensord::core::Engine<float>> engine,
             const std::vector<int> &in_shape);
  void Predict(const std::vector<cv::Mat> &requests,
               std::vector<ResponseMix> *responses);

 private:
  std::shared_ptr<tensord::core::Engine<float>> engine_;
  VecFloat in_data_;
  int batch_, in_num_, in_c_, in_h_, in_w_;

  const float THRESHOLD[8] = {1.0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.4, 1.0};
};

}  // namespace mix
}  // namespace tron
