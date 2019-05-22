#include "infer_implement.hpp"

namespace Tron {

void Classification::Setup(const tron::MetaNetParam &meta_net_param,
                           int gpu_id) {
  net_.Setup(gpu_id);

  net_.LoadModel(meta_net_param.network(0));

  const auto &in_blob = net_.in_blob();
  CHECK_EQ(in_blob.size(), 1);
  in_str_ = in_blob[0];

  const auto &out_blob = net_.out_blob();
  CHECK_EQ(out_blob.size(), 1);
  prob_str_ = out_blob[0];

  const auto &data_shape = net_.GetBlobShapeByName<float>(in_str_);
  CHECK_EQ(data_shape.size(), 4);

  batch_ = data_shape[0];
  in_c_ = data_shape[1];
  in_h_ = data_shape[2];
  in_w_ = data_shape[3];
  in_num_ = in_c_ * in_h_ * in_w_;

  in_data_.resize(batch_ * in_num_);

  num_classes_ = net_.get_single_argument<int>("num_classes", 1000);
}

void Classification::Predict(
    const cv::Mat &im_mat, const VecRectF &rois,
    std::vector<std::map<std::string, VecFloat>> *scores) {
  CHECK_LE(rois.size(), batch_);
  for (int b = 0; b < rois.size(); ++b) {
    ConvertData(im_mat, in_data_.data() + b * in_num_, rois[b], in_c_, in_h_,
                in_w_);
  }

  Process(in_data_, scores);

  CHECK_EQ(scores->size(), rois.size());
}

void Classification::Process(
    const VecFloat &in_data,
    std::vector<std::map<std::string, VecFloat>> *scores) {
  std::map<std::string, float *> data_map;
  data_map[in_str_] = const_cast<float *>(in_data.data());

  net_.Forward(data_map);

  const auto *prob_data = net_.GetBlobDataByName<float>(prob_str_);

  scores->clear();
  for (int b = 0; b < batch_; ++b) {
    std::map<std::string, VecFloat> score_map;
    score_map["score"] = VecFloat(prob_data, prob_data + num_classes_);
    scores->push_back(score_map);
    prob_data += num_classes_;
  }
}

}  // namespace Tron
