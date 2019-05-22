#include "inference_mix.hpp"

#include "glog/logging.h"

#include "common/image.hpp"

namespace tron {
namespace mix {

// 在nums[begin:end]的数组中，找到最大值的索引
static int GetIndexOfMaxNumber(float *nums, int begin, int end) {
  int ans = begin;
  float max = nums[begin];
  for (int i = begin; i < end; i++)
    if (nums[i] > max) {
      max = nums[i];
      ans = i;
    }
  return ans - begin;
}

void InferenceMix::Setup(
    std::shared_ptr<tensord::core::Engine<float>> engine,
    const std::vector<int> &in_shape) {
  CHECK_EQ(in_shape.size(), 4);
  batch_ = in_shape[0];
  in_c_ = in_shape[1], in_h_ = in_shape[2], in_w_ = in_shape[3];
  in_num_ = in_c_ * in_h_ * in_w_;
  in_data_.resize(batch_ * in_num_);

  engine_ = engine;
}

void InferenceMix::Predict(const std::vector<cv::Mat> &requests,
                           std::vector<ResponseMix> *responses) {
  std::vector<tensord::core::NetIn<float>> ins;
  for (std::size_t i = 0; i < requests.size(); ++i) {
    ConvertData(requests[i],
                in_data_.data() + i * in_num_,
                RectF(0, 0, requests[i].cols, requests[i].rows),
                in_c_, in_h_, in_w_);
    int n = in_h_ * in_w_;
    auto data = in_data_.data() + i * in_num_;
    for (int j = 0; j < n; j++) {
      *(data + n * 0 + j) = (*(data + n * 0 + j) - 103.52) * 0.017;
      *(data + n * 1 + j) = (*(data + n * 1 + j) - 116.28) * 0.017;
      *(data + n * 2 + j) = (*(data + n * 2 + j) - 123.675) * 0.017;
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
    auto w = requests[i].cols,
         h = requests[i].rows;
    auto out = outs[i];
    auto loc = out.GetByName("loc");
    auto prob_bk = out.GetByName("prob_bk");
    auto prob_pulp = out.GetByName("prob_pulp");

    ResponseMix resp;
    for (std::size_t j = 0; j < loc.size(); j += 7) {
      int label = loc[j + 1];
      if (label == -1) break;
      if (loc[j + 2] <= THRESHOLD[label]) continue;
      if (label != 6) {
        resp.bk = 1;
        continue;
      }
      resp.face = 1;
      resp.boxes.emplace_back(w * loc[j + 3],
                              h * loc[j + 4],
                              w * loc[j + 5],
                              h * loc[j + 6]);
    }
    switch (GetIndexOfMaxNumber(prob_pulp.data(), 0, prob_pulp.size())) {
      case 0:
      case 1: {
        resp.pulp = 1;
        break;
      }
    }
    auto bk = GetIndexOfMaxNumber(prob_bk.data(), 0, prob_bk.size());
    if (bk < 7 || (bk >= 9 && bk < 11) || (bk >= 12 && bk < 19)) {
      resp.bk = 1;
    } else if ((bk >= 7 && bk < 9) || (bk >= 42 && bk < 44)) {
      resp.march = 1;
    } else if (bk == 11 || (bk >= 28 && bk < 30)) {
      resp.text = 1;
    }
    responses->push_back(resp);
  }
}

}  // namespace mix
}  // namespace tron
