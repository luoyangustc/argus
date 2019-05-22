#ifndef TRON_TERROR_MIXUP_FORWARD
#define TRON_TERROR_MIXUP_FORWARD

#include "forward.hpp"
#include "forward.pb.h"
#include "framework/context.hpp"
#include "infer.hpp"
#include "inference.hpp"
#include "process.hpp"
#include "utils.hpp"

namespace tron {
namespace terror_mixup {

using std::string;
using std::vector;

vector<string> forward_custom_file_list = {
    "det_deploy.prototxt", "det_weight.caffemodel",
    "fine_deploy.prototxt", "fine_weight.caffemodel",
    "coarse_deploy.prototxt", "coarse_weight.caffemodel"};

class Forward : public tron::framework::ForwardBase<ForwardRequest,
                                                    ForwardResponse, Config> {
 public:
  Forward() = default;
  ~Forward() {}

  void Setup(const std::vector<std::vector<char>> &net_param_data,
             const VecInt &in_shape, const int &gpu_id,
             const Config &conf) override {
    batch_size = conf.batch_size;
    LOG(INFO) << "forward setup batch_size:" << batch_size
              << " gpu_id:" << gpu_id;
    terrorMixupDet = std::make_unique<TerrorMixupDet>(
        vector_char_to_string(net_param_data[0]),
        vector_char_to_string(net_param_data[1]), conf.batch_size);
    terrorMixupFine = std::make_unique<TerrorMixupFine>(
        vector_char_to_string(net_param_data[2]),
        vector_char_to_string(net_param_data[3]), conf.batch_size);
    terrorMixupCoarse = std::make_unique<TerrorMixupCoarse>(
        vector_char_to_string(net_param_data[4]),
        vector_char_to_string(net_param_data[5]), conf.batch_size);
    auto init = [gpu_id]() {
      caffe::Caffe::set_mode(caffe::Caffe::GPU);
      caffe::Caffe::SetDevice(gpu_id);
    };
    tron::framework::ForwardBase<ForwardRequest, ForwardResponse,
                                 Config>::Setup(in_shape, init);
  };
  void Release() override{};

 private:
  int batch_size = -1;
  std::unique_ptr<TerrorMixupDet> terrorMixupDet;
  std::unique_ptr<TerrorMixupFine> terrorMixupFine;
  std::unique_ptr<TerrorMixupCoarse> terrorMixupCoarse;
  void Process(
      std::vector<ForwardRequest>::const_iterator requests_first,
      std::vector<ForwardRequest>::const_iterator requests_last,
      std::vector<ForwardResponse>::iterator responses_first) override {
    vector<vector<float>> det_input_s;
    vector<vector<float>> cls_input_s;
    int i = 0;
    for (auto cur = requests_first; cur != requests_last; ++cur) {
      i++;
      vector<float> det_input;
      vector<float> cls_input;
      string_unmarshal_to_vector(const_cast<ForwardRequest &>(*cur).data().det_input(),
                                 det_input);
      string_unmarshal_to_vector(const_cast<ForwardRequest &>(*cur).data().cls_input(),
                                 cls_input);

      det_input_s.push_back(det_input);
      cls_input_s.push_back(cls_input);
    }

    const int size = i;
    LOG(INFO) << "size:" << size << " batch_size:" << batch_size;
    for (int j = size; j < batch_size; j++) {
      det_input_s.push_back(vector<float>(input_det_shape.single_batch_size()));
      cls_input_s.push_back(
          vector<float>(input_coarse_shape.single_batch_size()));
    }

    const auto det_input_batch = join_batch_size_data(det_input_s);
    const auto cls_input_batch = join_batch_size_data(cls_input_s);
    const auto output_det = terrorMixupDet->forward(det_input_batch);
    const auto output_fine = terrorMixupFine->forward(cls_input_batch);
    const auto output_coarse = terrorMixupCoarse->forward(cls_input_batch);
    for (int j = 0; j < size; ++j) {
      auto resp = responses_first + j;
      string output_coarse_s, output_fine_s, output_det_s;

      vector_marshal_to_string(output_coarse, output_coarse_s);
      vector_marshal_to_string(output_fine, output_fine_s);
      vector_marshal_to_string(output_det, output_det_s);

      resp->set_output_coarse(output_coarse_s);
      resp->set_output_fine(output_fine_s);
      resp->set_output_det(output_det_s);
      resp->set_batch_index(j);
    }
  };
};

}  // namespace terror_mixup
}  // namespace tron
#endif  // TRON_FACE_FEATURE_FORWARD_FEATURE_HPP NOLINT
