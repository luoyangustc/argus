#include "inference_qa.hpp"

#include <algorithm>
#include <utility>

#include "common/boxes.hpp"
#include "common/image.hpp"
#include "common/time.hpp"

namespace tron {
namespace fd {

void QualityInference::Setup(
    std::shared_ptr<tensord::core::Engine<float>> engine,
    const std::vector<int> &in_shape,
    const QualityInferenceConfig &conf) {
  CHECK_EQ(in_shape.size(), 4);
  batch_ = in_shape[0];
  in_c_ = in_shape[1], in_h_ = in_shape[2], in_w_ = in_shape[3];
  in_num_ = in_c_ * in_h_ * in_w_;
  in_data_.resize(batch_ * in_num_);

  conf_ = conf;
  engine_ = engine;
}

void QualityInference::Predict(const std::vector<QaRequest> &ins,
                               std::vector<QaResponse> *outs) {
  auto t1 = Time();
  size_t size = ins.size();
  in_data_.resize(size * in_num_);
  outs->resize(size);
  for (std::size_t i = 0; i < size; i++) {
    outs->at(i).probs.resize(cls_dim_);
    outs->at(i).label = -1;
    outs->at(i).orientation = -1;
  }

  std::vector<tensord::core::NetIn<float>> requests;

  // VecBoxF qa_boxes;
  float scale = 0.1;  // expend face box
  int index = -1;
  std::vector<int> indexes;
  for (const auto &in : ins) {
    auto &box = in.face_box;
    ++index;
    // filter < 48 small face
    int width = box.xmax - box.xmin + 1;
    int height = box.ymax - box.ymin + 1;
    if (width < in.min_face || height < in.min_face) {
      outs->at(index).label = 5;  // index 5->"small"
      continue;
    }
    auto im_mat = in.im_mat;

    BoxF exbox;
    exbox.xmin = std::max(0, cvRound(box.xmin - scale * width));
    exbox.ymin = std::max(0, cvRound(box.ymin - scale * height));
    exbox.xmax = std::min(im_mat.cols - 1, cvRound(box.xmax + scale * width));
    exbox.ymax = std::min(im_mat.rows - 1, cvRound(box.ymax + scale * height));

    ConvertData(im_mat, in_data_.data() + indexes.size() * in_num_,
                exbox.RectFloat(), in_c_, in_h_, in_w_);

    tensord::core::NetIn<float> request;
    request.names.push_back("data");
    request.datas.emplace_back(
        in_data_.begin() + indexes.size() * in_num_,
        in_data_.begin() + indexes.size() * in_num_ + in_num_);
    requests.push_back(request);
    indexes.push_back(index);
  }
  if (indexes.size() == 0) {
    return;
  }

  std::vector<tensord::core::NetOut<float>> responses;
  auto t2 = Time();
  engine_->Predict(requests, &responses);
  auto t3 = Time();

  for (std::size_t i = 0; i < indexes.size(); i++) {
    auto resp = responses[i];
    outs->at(indexes[i]).probs.clear();

    float *cls_ptr = &resp.GetByName("softmax")[0];
    float *ori_ptr = &resp.GetByName("pose_softmax")[0];
    int label, orientation;
    float conf;
    float orient_conf;
    Argmax(cls_ptr, cls_dim_, &label, &conf);
    Argmax(ori_ptr, ori_dim_, &orientation, &orient_conf);
    outs->at(indexes[i]).orientation = orientation;
    switch (label) {
      case 0:  // quality
        if (conf > this->conf_.quality_threshold) {
          outs->at(indexes[i]).label = 0;
        } else {
          outs->at(indexes[i]).label = 2;
        }
        break;
      case 1:  // neg
        outs->at(indexes[i]).label = 1;
        break;
      case 2:  // blur
        outs->at(indexes[i]).label = 2;
        break;
      case 3:  // pose
        outs->at(indexes[i]).label = 3;
        break;
      case 4:  // cover
        outs->at(indexes[i]).label = 4;
        break;
      default: {}
    }
    switch (label) {
      case 0:
      case 2:
      case 3:
      case 4:
        for (int j = 0; j < cls_dim_; j++) {
          outs->at(indexes[i]).probs.push_back(*(cls_ptr + j));
        }
        break;
      default: {}
    }
  }
  auto t4 = Time();
  LOG(INFO) << t2.since_millisecond(t1) << " "
            << t3.since_millisecond(t2) << " " << t4.since_millisecond(t3);
}

void QualityInference::Argmax(const float *data, const int num, int *idx,
                              float *conf) {
  *conf = data[0];
  *idx = 0;
  for (int iter = 1; iter < num; iter++) {
    if (data[iter] > *conf) {
      *idx = iter;
      *conf = data[iter];
    }
  }
}

}  // namespace fd
}  // namespace tron
