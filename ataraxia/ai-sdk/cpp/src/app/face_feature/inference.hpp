#ifndef TRON_FACE_FEATURE_INFERENCE_HPP  // NOLINT
#define TRON_FACE_FEATURE_INFERENCE_HPP

#include <vector>

#include <opencv2/opencv.hpp>

#include "feature/inference_feature.hpp"
#include "mtcnn/inference_mtcnn.hpp"

namespace tron {
namespace ff {

struct Inference {
 public:
  void Predict(const std::vector<cv::Mat> &im_mats,
               const std::vector<BoxF> &boxes,
               const std::vector<int> &mirror_tricks,
               std::vector<std::vector<float>> *features);

  int min_face_size_ = 20;
  int mirror_trick_ = 0;

  std::shared_ptr<FeatureInference> feature_;
  std::shared_ptr<MTCNNInference> mtcnn_;
};

}  // namespace ff
}  // namespace tron

#endif  // TRON_FACE_FEATURE_INFERENCE_HPP NOLINT