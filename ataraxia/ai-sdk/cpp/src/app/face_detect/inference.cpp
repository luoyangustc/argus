
#include "inference.hpp"

#include "common/errors.hpp"
#include "common/json.hpp"
#include "common/time.hpp"

namespace tron {
namespace fd {

void Inference::Predict(const cv::Mat &im_mat,
                        TronDetectionOutput *output,
                        bool use_quality) {
  int im_h = im_mat.rows, im_w = im_mat.cols;

  auto t1 = Time();

  // if (im_h <= 1 || im_w <= 1) {
  //   return tron::tron_status_image_size_error;
  // }
  // detect out
  std::vector<tron::VecBoxF> Gboxes;
  std::vector<FDRequest> rois{{im_mat, {tron::RectF(0, 0, im_w, im_h)}}};
  fd_->Predict(rois, &Gboxes);
  output->objects.clear();
  // quality
  std::vector<QaRequest> qaReqs;
  std::vector<QaResponse> qaResps;
  Time t2 = Time();
  if (use_quality && Gboxes[0].size() > 0) {
    if (qa_ == nullptr) {
      LOG(WARNING) << "Tron quality is uninitialized";
      // return tron_status_method_nullptr;
    }
    qaReqs.reserve(Gboxes[0].size());
    for (std::size_t i = 0; i < Gboxes[0].size(); i++) {
      qaReqs.emplace_back(im_mat, Gboxes[0][i], min_face);
    }
    t2 = Time();
    qa_->Predict(qaReqs, &qaResps);
  }
  auto t3 = Time();

  tron::VecBoxF boxes;
  boxes = Gboxes[0];

  for (std::size_t iter = 0; iter < boxes.size(); iter++) {
    if (!use_quality) {  // only facedetection
      const auto box = boxes[iter];
      TronRectangle rect = {};
      rect.xmin = static_cast<int>(box.xmin);
      rect.xmax = static_cast<int>(box.xmax);
      rect.ymin = static_cast<int>(box.ymin);
      rect.ymax = static_cast<int>(box.ymax);
      rect.id = box.label;
      rect.score = box.score;
      if (box.label >= 0 && box.label < static_cast<int>(labels_.size())) {
        rect.label = labels_[box.label];
      } else {
        rect.label = "";
      }
      output->objects.push_back(rect);
    } else {
      if (qaResps[iter].label != 1) {  // not neg face
        const auto box = boxes[iter];
        TronRectangle rect = {};
        rect.xmin = static_cast<int>(box.xmin);
        rect.xmax = static_cast<int>(box.xmax);
        rect.ymin = static_cast<int>(box.ymin);
        rect.ymax = static_cast<int>(box.ymax);
        rect.id = box.label;
        rect.score = box.score;
        rect.quality_category = -1;
        if (box.label >= 0 &&
            static_cast<size_t>(box.label) < this->labels_.size()) {
          rect.label = this->labels_[box.label];
        } else {
          rect.label = "";
        }
        if (!this->output_quality_score) {
          rect.quality_category = qaResps[iter].label;
          for (int i = 0; i < 5; ++i) {
            rect.quality_cls[i] = -1;
          }
        } else {
          const auto &quality_prob = qaResps[iter].probs;
          rect.quality_category = qaResps[iter].label;
          for (int i = 0; i < 5; ++i) {
            rect.quality_cls[i] = quality_prob[i];
          }
        }
        rect.orient_category = qaResps[iter].orientation;
        output->objects.push_back(rect);
      }  // if(Qalabels[iter]!=1)
    }
  }

  auto t4 = Time();
  LOG(INFO) << t2.since_millisecond(t1) << " "
            << t3.since_millisecond(t2) << " "
            << t4.since_millisecond(t3);
  return;
}

}  // namespace fd
}  // namespace tron
