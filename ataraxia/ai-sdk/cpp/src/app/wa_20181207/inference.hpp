#ifndef TRON_FACE_FEATURE_INFERENCE_HPP
#define TRON_FACE_FEATURE_INFERENCE_HPP

#include <map>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/opencv.hpp>

#include "forward.hpp"
#include "framework/forward.hpp"

namespace tron {
namespace wa {

struct DetectionResult {
  DetectionResult() {}
  DetectionResult(int index, float score, std::string clas, int pts[4])
      : index(index), score(score), clas(clas) {
    this->pts[0] = pts[0];
    this->pts[1] = pts[1];
    this->pts[2] = pts[2];
    this->pts[3] = pts[3];
  }
  ~DetectionResult() {}

  int index;
  double score;
  std::string clas;
  int pts[4];
};

struct ConfidenceResult {
  ConfidenceResult() {}
  ConfidenceResult(int index, float score, std::string clas)
      : index(index), score(score), clas(clas) {}
  ~ConfidenceResult() {}

  int index;
  double score;
  std::string clas;
};

class Result {
 public:
  std::vector<DetectionResult> detections;
  std::vector<ConfidenceResult> confidences;
};

struct InferenceConfig {
  InferenceConfig() = default;
  InferenceConfig(std::map<int, std::pair<std::string, float>> labels_det,
                  std::map<int, std::string> labels_fine)
      : labels_det(labels_det), labels_fine(labels_fine) {}
  ~InferenceConfig() {}

  std::map<int, std::pair<std::string, float>> labels_det;
  std::map<int, std::string> labels_fine;
};

class Inference
    : public framework::InferenceBase<
          framework::ForwardWrap<Forward,
                                 inference::wa::ForwardRequest,
                                 inference::wa::ForwardResponse>,
          cv::Mat,
          Result,
          InferenceConfig> {
 public:
  void Predict(const std::vector<cv::Mat> &im_mats,
               std::vector<Result> *results) override;
};

}  // namespace wa
}  // namespace tron

#endif  // TRON_FACE_FEATURE_INFERENCE_HPP NOLINT