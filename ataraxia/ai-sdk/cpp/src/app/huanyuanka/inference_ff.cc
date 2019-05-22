#include "inference_ff.hpp"

#include "glog/logging.h"

#include "common/image.hpp"

namespace tron {
namespace mix {

void InferenceFF::Setup(
    std::shared_ptr<tensord::core::Engine<float>> engine,
    const std::vector<int> &in_shape) {
  CHECK_EQ(in_shape.size(), 4);
  batch_ = in_shape[0];
  in_c_ = in_shape[1], in_h_ = in_shape[2], in_w_ = in_shape[3];
  in_num_ = in_c_ * in_h_ * in_w_;
  in_data_.resize(batch_ * in_num_);

  engine_ = engine;
}

void InferenceFF::Predict(const std::vector<cv::Mat> &requests,
                          std::vector<std::vector<float>> *responses) {
  std::vector<tensord::core::NetIn<float>> ins;
  for (std::size_t i = 0; i < requests.size(); ++i) {
    int w = requests[i].cols, h = requests[i].rows;
    ConvertData(requests[i],
                in_data_.data() + i * in_num_,
                RectF(0, 0, w, h),
                in_c_, in_h_, in_w_);
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
    auto feature = out.GetByName("fc");
    responses->push_back(feature);
  }
}

}  // namespace mix
}  // namespace tron
