#ifndef TRON_TERROR_MIXUP_INFERENCE
#define TRON_TERROR_MIXUP_INFERENCE

#include <opencv2/opencv.hpp>
#include "forward.hpp"
#include "forward.pb.h"
#include "framework/context.hpp"
#include "post_process.hpp"
#include "pre_process.hpp"

namespace tron {
namespace terror_mixup {

class Forward;

vector<string> inference_custom_file_list = {
    "fine_labels.csv", "det_labels.csv", "coarse_labels.csv"};
class Inference
    : public tron::framework::InferenceBase<
          framework::ForwardWrap<Forward, ForwardRequest, ForwardResponse>,
          cv::Mat, string, Config> {
  using Request = ForwardRequest;
  using Response = ForwardResponse;
  using ForwardWrap = framework::ForwardWrap<Forward, Request, Response>;
  using Base =
      tron::framework::InferenceBase<ForwardWrap, cv::Mat, string, Config>;

 public:
  Inference() = default;
  ~Inference() {}

  void Setup(const std::vector<int> &in_shape,
             std::shared_ptr<ForwardWrap> forward, const Config &config) {
    batch_size = config.batch_size;

    CHECK_EQ(config.params.files.size(), inference_custom_file_list.size());
    post_process_param param;
    {
      param.fine_labels = config.params.files[0];
      param.det_labels = config.params.files[1];
      param.coarse_labels = config.params.files[2];
      param.percent_fine = 0.6;
      param.percent_coarse = 0.4;
      param.batch_size = batch_size;
    }
    post_process = std::make_unique<PostProcess>(param);
    Base::Setup(in_shape, forward, config);
  }

  void Predict(const std::vector<cv::Mat> &im_mats,
               std::vector<string> *outs) override {
    std::vector<Request> requests(im_mats.size());
    std::vector<Response> responses(im_mats.size());

    vector<int> image_widths;
    vector<int> image_heights;
    for (std::size_t batch_index = 0; batch_index < im_mats.size();
         ++batch_index) {
      const auto im_ori = im_mats[batch_index];
      const auto image_width = im_ori.size().width;
      const auto image_height = im_ori.size().height;
      image_widths.push_back(image_width);
      image_heights.push_back(image_height);
      const vector<float> det_input = det_pre_process_image(im_ori);
      const vector<float> cls_input = cls_pre_process_image(im_ori);
      string det_input_s, cls_input_s;
      vector_marshal_to_string(det_input, det_input_s);
      vector_marshal_to_string(cls_input, cls_input_s);

      requests[batch_index].mutable_data()->set_det_input(det_input_s);
      requests[batch_index].mutable_data()->set_cls_input(cls_input_s);
    }
    forward_->Predict(requests, &responses);
    outs->clear();
    for (std::size_t batch_index = 0; batch_index < im_mats.size();
         ++batch_index) {
      vector<float> output_coarse, output_fine, output_det;
      string_unmarshal_to_vector(responses[batch_index].output_coarse(), output_coarse);
      string_unmarshal_to_vector(responses[batch_index].output_fine(), output_fine);
      string_unmarshal_to_vector(responses[batch_index].output_det(), output_det);
      auto response = post_process->process(
          output_fine, output_coarse, output_det,
          responses[batch_index].batch_index(), image_widths[batch_index],
          image_heights[batch_index]);
      auto text = dump_msg(response);
      outs->push_back(text);
    }
  };

 private:
  std::unique_ptr<PostProcess> post_process;
  int batch_size = -1;
};

}  // namespace terror_mixup
}  // namespace tron
#endif  // TRON_FACE_FEATURE_FORWARD_FEATURE_HPP NOLINT
