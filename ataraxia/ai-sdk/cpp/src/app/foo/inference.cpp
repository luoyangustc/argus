#include "inference.hpp"

#include "glog/logging.h"

#include "common/image.hpp"
#include "common/time.hpp"

namespace tron {
namespace foo {

void Inference::Predict(const std::vector<cv::Mat> &im_mats,
                        std::vector<float> *outs) {
  in_data_.resize(im_mats.size() * in_num_);

  std::vector<Request> requests;
  std::vector<Response> responses;

  auto t1 = Time();
  for (std::size_t i = 0; i < im_mats.size(); i++) {
    // cv::imwrite("/src/res/tmp/0.jpg", im_mats[i]);
    ConvertData(im_mats[i], in_data_.data() + i * in_num_,
                RectF(0, 0, im_mats[i].cols, im_mats[i].rows),
                in_c_, in_h_, in_w_);
    Request request;
    request.mutable_data()->mutable_body()->assign(
        reinterpret_cast<const char *>(&in_data_[i * in_num_]),
        in_num_ * sizeof(float));
    requests.push_back(request);
    responses.push_back(Response());
  }
  auto t2 = Time();

  forward_->Predict(requests, &responses);

  auto t3 = Time();
  LOG(INFO) << t2.since_millisecond(t1) << " " << t3.since_millisecond(t2);
  outs->clear();
  for (std::size_t i = 0; i < im_mats.size(); i++) {
    auto resp = responses[i];
    outs->push_back(resp.sum());
  }
}

}  // namespace foo
}  // namespace tron
