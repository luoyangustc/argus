#ifndef TRON_FACE_DETECT_INFERENCE_QA_HPP
#define TRON_FACE_DETECT_INFERENCE_QA_HPP

#include <vector>

#include <opencv2/opencv.hpp>

#include "common/type.hpp"
#include "tensord/tensord.hpp"

namespace tron {
namespace fd {

class QualityInferenceConfig {
 public:
  int gpu_id;
  VecInt in_shape;
  float neg_threshold, pose_threshold, cover_threshold, blur_threshold,
      quality_threshold;
};

struct QaRequest {
  QaRequest() = default;
  QaRequest(cv::Mat im_mat, const BoxF face_box, const int min_face)
      : im_mat(im_mat), face_box(face_box), min_face(min_face) {}
  ~QaRequest() {}

  cv::Mat im_mat;
  BoxF face_box;
  int min_face;
};

struct QaResponse {
  QaResponse() = default;
  QaResponse(std::vector<float> probs, int label, int orientation)
      : probs(probs), label(label), orientation(orientation) {}
  ~QaResponse() {}

  std::vector<float> probs;
  int label;
  int orientation;
};

class QualityInference {
 public:
  QualityInference() = default;
  ~QualityInference() {}

  void Setup(std::shared_ptr<tensord::core::Engine<float>> engine,
             const std::vector<int> &in_shape,
             const QualityInferenceConfig &conf);
  void Predict(const std::vector<QaRequest> &ins,
               std::vector<QaResponse> *outs);

 private:
  void Argmax(const float *data, const int num, int *idx, float *conf);

  std::shared_ptr<tensord::core::Engine<float>> engine_;
  VecFloat in_data_;
  int batch_, in_num_, in_c_, in_h_, in_w_;

  int cls_dim_ = 5;
  int ori_dim_ = 8;
  QualityInferenceConfig conf_;
};

}  // namespace fd
}  // namespace tron

#endif  // TRON_FACE_DETECT_INFERENCE_QA_HPP NOLINT
