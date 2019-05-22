#ifndef TRON_FACE_DETECT_INFERENCE_FD_HPP
#define TRON_FACE_DETECT_INFERENCE_FD_HPP

#include <vector>

#include <opencv2/opencv.hpp>
#include "glog/logging.h"

#include "common/type.hpp"
#include "proto/tron.pb.h"
#include "tensord/tensord.hpp"

namespace tron {
namespace fd {

struct FDRequest {
  FDRequest() = default;
  FDRequest(cv::Mat im_mat, const RectF &roi) : im_mat(im_mat), roi(roi) {}
  ~FDRequest() {}

  cv::Mat im_mat;
  RectF roi;
};

class FDInference {
 public:
  FDInference() = default;
  ~FDInference() {}

  void Setup(std::shared_ptr<tensord::core::Engine<float>> engine,
             const std::vector<int> &in_shape);
  void Predict(const std::vector<FDRequest> &ins,
               std::vector<VecBoxF> *outs);

 private:
  using LabelBBox = std::map<int, VecBoxF>;
  using VecLabelBBox = std::vector<LabelBBox>;

  std::shared_ptr<tensord::core::Engine<float>> engine_;
  VecFloat in_data_;
  int batch_, in_num_, in_c_, in_h_, in_w_;

  float threshold_ = 0.5;
  int top_k_ = 500;
  int keep_top_k_ = 500;
  float nms_threshold_ = 0.3;
  float confidence_threshold_ = 0.3;
};

}  // namespace fd
}  // namespace tron

#endif  // TRON_FACE_DETECT_INFERENCE_FD_HPP NOLINT
