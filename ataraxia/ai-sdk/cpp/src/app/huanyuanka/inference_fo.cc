#include "inference_fo.hpp"

#include "glog/logging.h"

#include "common/image.hpp"
#include "face_alignment.hpp"

namespace tron {
namespace mix {

void InferenceFO::Setup(
    std::shared_ptr<tensord::core::Engine<float>> engine,
    const std::vector<int> &in_shape) {
  CHECK_EQ(in_shape.size(), 4);
  batch_ = in_shape[0];
  in_c_ = in_shape[1], in_h_ = in_shape[2], in_w_ = in_shape[3];
  in_num_ = in_c_ * in_h_ * in_w_;
  in_data_.resize(batch_ * in_num_);

  engine_ = engine;
}

void InferenceFO::Predict(const std::vector<RequestFO> &requests,
                          std::vector<cv::Mat> *responses) {
  VecBoxF boxes;
  std::vector<tensord::core::NetIn<float>> ins;
  for (std::size_t i = 0; i < requests.size(); ++i) {
    int iw = requests[i].im.cols, ih = requests[i].im.rows;
    int w = requests[i].box.xmax - requests[i].box.xmin + 1,
        h = requests[i].box.ymax - requests[i].box.ymin + 1;
    BoxF box(std::max(0.0, requests[i].box.xmin - 0.0143 * w),
             std::max(0.0, requests[i].box.ymin - 0.0143 * h),
             std::min(static_cast<double>(iw - 1),
                      requests[i].box.xmax + 0.0143 * w),
             std::min(static_cast<double>(ih - 1),
                      requests[i].box.ymax + 0.0143 * h));
    boxes.push_back(box);
    ConvertData(requests[i].im,
                in_data_.data() + i * in_num_,
                RectF(box.xmin, box.ymin,
                      box.xmax - box.xmin + 1,
                      box.ymax - box.ymin + 1),
                in_c_, in_h_, in_w_,
                1, true);
    int n = in_h_ * in_w_;
    auto data = in_data_.data() + i * in_num_;
    for (int j = 0; j < n; j++) {
      *(data + n * 0 + j) = (*(data + n * 0 + j) - 127.5) / 128;
      *(data + n * 1 + j) = (*(data + n * 1 + j) - 127.5) / 128;
      *(data + n * 2 + j) = (*(data + n * 2 + j) - 127.5) / 128;
    }
    tensord::core::NetIn<float> in;
    in.names.push_back("data");
    in.datas.emplace_back(in_data_.begin() + i * in_num_,
                          in_data_.begin() + i * in_num_ + in_num_);
    ins.push_back(in);
  }

  std::vector<tensord::core::NetOut<float>> outs;
  engine_->Predict(ins, &outs);

  for (std::size_t i = 0; i < outs.size(); i++) {
    auto out = outs[i];
    auto point = out.GetByName("conv");
    auto conf = out.GetByName("prob");

    std::vector<cv::Point2d> points(5);
    for (int j = 0; j < 5; j++) {
      points[j] = cv::Point2d(
          point[j] * (boxes[i].xmax - boxes[i].xmin) + boxes[i].xmin,
          point[j + 5] * (boxes[i].ymax - boxes[i].ymin) + boxes[i].ymin);
    }

    cv::Mat face;
    if (conf[1] >= 0.7) {
      faceAlignmet(requests[i].im, points, face);
    }
    responses->push_back(face);
  }
}

}  // namespace mix
}  // namespace tron
