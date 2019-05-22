#include "inference_feature.hpp"

#include "glog/logging.h"

#include "common/image.hpp"

namespace tron {
namespace ff {

void FeatureInference::Setup(
    std::shared_ptr<tensord::core::Engine<float>> engine,
    const std::vector<int> &in_shape,
    const FeatureConfig &config) {
  CHECK_EQ(in_shape.size(), 4);
  batch_ = in_shape[0];
  in_c_ = in_shape[1], in_h_ = in_shape[2], in_w_ = in_shape[3];
  in_num_ = in_c_ * in_h_ * in_w_;
  in_data_.resize(batch_ * in_num_);

  config_ = config;
  engine_ = engine;
}

void FeatureInference::Predict(const std::vector<FeatureRequest> &requests,
                               std::vector<std::vector<float>> *features) {
  LOG(INFO) << "begin...";
  in_data_.resize(requests.size() * in_num_ * 2);

  std::vector<tensord::core::NetIn<float>> ins;

  int index = 0;
  for (std::size_t i = 0; i < requests.size(); i++) {
    // cv::imwrite("/src/res/tmp/0.jpg", im_mats[i]);
    ConvertData(requests[i].im_mat, in_data_.data() + index * in_num_,
                requests[i].roi, in_c_, in_h_, in_w_);
    tensord::core::NetIn<float> in;
    in.names.push_back("data");
    in.datas.emplace_back(in_data_.begin() + index * in_num_,
                          in_data_.begin() + index * in_num_ + in_num_);
    ins.push_back(in);
    index++;

    if (requests[i].mirror_trick == 1) {
      cv::Mat im_mat_flip = requests[i].im_mat.clone();
      cv::flip(requests[i].im_mat, im_mat_flip, 1);
      ConvertData(
          im_mat_flip, in_data_.data() + index * in_num_,
          RectF(im_mat_flip.cols - requests[i].roi.x - requests[i].roi.w,
                requests[i].roi.y,
                requests[i].roi.w,
                requests[i].roi.h),
          in_c_, in_h_, in_w_);
      tensord::core::NetIn<float> in2;
      in2.names.push_back("data");
      in2.datas.emplace_back(in_data_.begin() + index * in_num_,
                             in_data_.begin() + index * in_num_ + in_num_);
      ins.push_back(in2);
      index++;
    }
  }

  std::vector<tensord::core::NetOut<float>> outs;
  engine_->Predict(ins, &outs);

  features->clear();
  index = 0;
  for (std::size_t i = 0; i < requests.size(); i++) {
    auto out = outs[index];
    std::vector<float> feature(out.datas[0]);
    index++;

    if (requests[i].mirror_trick == 1) {
      auto out1 = outs[index];
      std::vector<float> feature1(out1.datas[0]);
      int j = 0;
      for (auto cur = feature1.begin(); cur != feature1.end(); cur++) {
        feature[j] = (feature[j] + *cur) / 2.0f;
        j++;
      }
      index++;
    }

    float sum = 0;
    for (auto cur = feature.begin(); cur != feature.end(); cur++) {
      sum += *cur * *cur;
    }
    sum = std::sqrt(sum);
    for (auto cur = feature.begin(); cur != feature.end(); cur++) {
      *cur /= sum;
    }

    features->push_back(feature);
  }
}

}  // namespace ff
}  // namespace tron
