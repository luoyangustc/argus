#ifndef TRON_FACE_FEATURE_INFERENCE_FEATURE_HPP  // NOLINT
#define TRON_FACE_FEATURE_INFERENCE_FEATURE_HPP

#include <vector>

#include <opencv2/opencv.hpp>

#include "common/boxes.hpp"
#include "common/type.hpp"
#include "tensord/tensord.hpp"

namespace tron {
namespace ff {

struct FeatureConfig {
  FeatureConfig() = default;
  explicit FeatureConfig(const std::string output_layer)
      : output_layer_(output_layer) {}

  std::string output_layer_;
};

struct FeatureRequest {
  FeatureRequest() = default;
  FeatureRequest(cv::Mat im_mat, RectF roi, int mirror_trick)
      : im_mat(im_mat), roi(roi), mirror_trick(mirror_trick) {}
  ~FeatureRequest() {}

  cv::Mat im_mat;
  RectF roi;
  int mirror_trick;
};

class FeatureInference {
 public:
  FeatureInference() = default;
  ~FeatureInference() {}

  void Setup(std::shared_ptr<tensord::core::Engine<float>> engine,
             const std::vector<int> &in_shape,
             const FeatureConfig &);
  void Predict(const std::vector<FeatureRequest> &,
               std::vector<std::vector<float>> *);

 private:
  std::shared_ptr<tensord::core::Engine<float>> engine_;
  FeatureConfig config_;
  VecFloat in_data_;
  int batch_, in_num_, in_c_, in_h_, in_w_;
};

}  // namespace ff
}  // namespace tron

#endif  // TRON_FACE_FEATURE_INFERENCE_FEATURE_HPP NOLINT
