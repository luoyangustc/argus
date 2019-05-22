#ifndef TRON_FACE_DETECT_INFERENCE_HPP
#define TRON_FACE_DETECT_INFERENCE_HPP

#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "const.hpp"
#include "inference_fd.hpp"
#include "inference_qa.hpp"

namespace tron {
namespace fd {

struct TronRectangle {
  /** Rectangle location and label index. */
  int xmin, ymin, xmax, ymax, id;
  /** Rectangle score. */
  float score;
  /** Rectangle label. */
  std::string label;
  /* quality cls probability*/
  float quality_cls[5];
  /* face orientation cls probability*/
  // float  orientation[8];
  int quality_category;
  int orient_category;
  std::string s_quality_category;
  std::string s_orient_category;
};

template <typename Archiver>
Archiver &operator&(Archiver &ar, TronRectangle &detection) {  // NOLINT
  ar.StartObject();
  ar.Member("index") & detection.id;
  ar.Member("score") & detection.score;
  ar.Member("class") & "face";
  ar.Member("score") & detection.score;

  ar.Member("pts");
  ar.StartArray();
  ar.StartArray() & detection.xmin &detection.ymin;
  ar.EndArray();
  ar.StartArray() & detection.xmax &detection.ymin;
  ar.EndArray();
  ar.StartArray() & detection.xmin &detection.ymax;
  ar.EndArray();
  ar.StartArray() & detection.xmin &detection.ymax;
  ar.EndArray();
  ar.EndArray();

  if (!detection.s_quality_category.empty() || ar.HasMember("quality")) {
    ar.Member("quality") & detection.s_quality_category;
  }
  if (!detection.s_orient_category.empty() || ar.HasMember("orientation")) {
    ar.Member("orientation") & detection.s_orient_category;
  }

  if ((detection.quality_category != 1 && detection.quality_category != 5) ||
      ar.HasMember("q_score")) {
    ar.Member("q_score");
    ar.StartObject();
    ar.Member("clear") & detection.quality_cls[0];
    ar.Member("blur") & detection.quality_cls[2];
    ar.Member("neg") & detection.quality_cls[1];
    ar.Member("cover") & detection.quality_cls[4];
    ar.Member("pose") & detection.quality_cls[3];
    ar.EndObject();
  }

  return ar.EndObject();
}

struct TronDetectionOutput {
  /** All detected objects. */
  std::vector<TronRectangle> objects;
};

struct Inference {
 public:
  void Predict(const cv::Mat &im_mat, TronDetectionOutput *output,
               bool use_quality);

  std::shared_ptr<FDInference> fd_;
  std::shared_ptr<QualityInference> qa_;

  // configuration
  std::vector<std::string> labels_;
  int batch_size = MAX_BATCH_SIZE;

  bool const_use_quality = true;
  float neg_threshold = 0;
  float pose_threshold = 0;
  float cover_threshold = 0;
  float blur_threshold = 0.98;
  float quality_threshold = 0.6;
  bool output_quality_score = false;
  int min_face = 50;
};

}  // namespace fd
}  // namespace tron

#endif  // TRON_FACE_DETECT_INFERENCE_HPP NOLINT