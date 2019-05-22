#include "inference.hpp"

#include <opencv2/opencv.hpp>
#include "glog/logging.h"

#include "common/image.hpp"

namespace tron {
namespace wa {

void Inference::Predict(const std::vector<cv::Mat> &im_mats,
                        std::vector<Result> *results) {
  std::vector<inference::wa::ForwardRequest> requests;
  std::vector<inference::wa::ForwardResponse> responses;

  std::vector<std::vector<float>> datas;
  for (std::size_t i = 0; i < im_mats.size(); i++) {
    int h = im_mats[i].rows, w = im_mats[i].cols;
    std::vector<float> data(w * h * 3);
    ConvertData(im_mats[i], data.data(),
                {0, 0, static_cast<float>(w), static_cast<float>(h)},
                3, h, w);
    inference::wa::ForwardRequest request;
    request.mutable_data()->mutable_body()->assign(
        reinterpret_cast<const char *>(data.data()),
        data.size() * sizeof(float));
    request.set_h(h);
    request.set_w(w);
    requests.push_back(request);
    responses.push_back(inference::wa::ForwardResponse());

    datas.push_back(data);
  }

  forward_->Predict(requests, &responses);

  results->clear();
  for (std::size_t i = 0; i < im_mats.size(); i++) {
    auto &im_mat = im_mats[i];
    auto &resp = responses[i];

    Result result;
    for (int j = 0; j < resp.label_size(); j++) {
      auto label = resp.label(j);
      std::string cls =
          label.score() < 0.9 ? "normal" : config_.labels_fine[label.index()];
      result.confidences.emplace_back(label.index(), label.score(), cls);
    }
    for (int j = 0; j < resp.boxes_size(); j++) {
      auto box = resp.boxes(j);
      if (box.score() < config_.labels_det[box.label()].second) {
        continue;
      }
      int pts[4] =
          {static_cast<int>(box.xmin() < 0.0 ? 0 : box.xmin() * im_mat.cols),
           static_cast<int>(box.ymin() < 0.0 ? 0 : box.ymin() * im_mat.rows),
           static_cast<int>(
               box.xmax() > 1.0 ? im_mat.cols : box.xmax() * im_mat.cols),
           static_cast<int>(
               box.ymax() > 1.0 ? im_mat.rows : box.ymax() * im_mat.rows)};
      LOG(INFO) << pts[0] << " " << pts[1] << " "
                << pts[2] << " " << pts[3];
      result.detections.emplace_back(
          static_cast<int>(box.label()), box.score(),
          config_.labels_det[box.label()].first, pts);
    }

    results->push_back(result);
  }
}

}  // namespace wa
}  // namespace tron
