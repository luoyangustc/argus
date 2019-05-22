#ifndef TRON_TERROR_MIXUP_PROCESS
#define TRON_TERROR_MIXUP_PROCESS

#include <string>
#include <utils.hpp>
#include <vector>
#include "./forward.pb.h"
#include "./post_process.hpp"
#include "./utils.hpp"
#include "caffe/caffe.hpp"
#include "debug_print.hpp"

namespace tron {
namespace terror_mixup {

std::unique_ptr<caffe::Net<float>> caffe_make_net(
    const caffe::NetParameter &net_param,
    const caffe::NetParameter &model_param);

std::unique_ptr<caffe::Net<float>> caffe_make_net_from_file_content(
    const string &prototext, const string &caffemodel, const int batch_size);

class TerrorMixupDet {
 public:
  TerrorMixupDet(const string &prototext, const string &caffemodel,
                 const int batch_size)
      : batch_size(batch_size) {
    net = caffe_make_net_from_file_content(prototext, caffemodel, batch_size);
    auto data_shape = net->blob_by_name("data")->shape();
    CHECK_EQ(data_shape[0] * data_shape[1] * data_shape[2] * data_shape[3],
             input_det_shape.single_batch_size() * batch_size);
    _DEBUG_PRINT(batch_size, net->blob_by_name("data")->shape(),
                 net->blob_by_name("detection_out")->shape());
  }
  std::unique_ptr<caffe::Net<float>> net;
  int batch_size;
  vector<float> forward(const vector<float> &det_input_batch) {
    input_det_shape.assert_shape_match(det_input_batch, batch_size);
    auto blob_data = net->blob_by_name("data")->data();
    CHECK_EQ(blob_data->size(), det_input_batch.size() * sizeof(float));
    memcpy(blob_data->mutable_cpu_data(), det_input_batch.data(),
           blob_data->size());
    net->Forward();
    vector<float> output_det(
        output_det_shape.single_batch_size());  // 固定，和batch_size无关
    auto detection_data = net->blob_by_name("detection_out")->data();
    CHECK_GE(output_det.size() * sizeof(float), detection_data->size());
    memcpy(output_det.data(), detection_data->cpu_data(),
           detection_data->size());
    memset(detection_data->mutable_cpu_data(), 0, detection_data->size());
    // 这里如果不清空的话，下次的推理结果会有脏数据，可能和 RefineDet 实现有关
    return output_det;
  }
};

class TerrorMixupFine {
 public:
  TerrorMixupFine(const string &prototext, const string &caffemodel,
                  const int batch_size)
      : batch_size(batch_size) {
    net = caffe_make_net_from_file_content(prototext, caffemodel, batch_size);
    _DEBUG_PRINT(batch_size, net->blob_by_name("data")->shape(),
                 net->blob_by_name("prob")->shape());
  }
  std::unique_ptr<caffe::Net<float>> net;
  int batch_size;
  vector<float> forward(const vector<float> &cls_input_batch) {
    input_fine_shape.assert_shape_match(cls_input_batch, batch_size);
    auto blob_data = net->blob_by_name("data")->data();
    CHECK_EQ(blob_data->size(),
             input_fine_shape.single_batch_size() * sizeof(float) * batch_size);
    memcpy(blob_data->mutable_cpu_data(), cls_input_batch.data(),
           blob_data->size());
    net->Forward();
    vector<float> output_fine(output_fine_shape.single_batch_size() *
                              batch_size);
    auto prob_data = net->blob_by_name("prob")->data();
    memcpy(output_fine.data(), prob_data->cpu_data(), prob_data->size());
    return output_fine;
  }
};

class TerrorMixupCoarse {
 public:
  TerrorMixupCoarse(const string &prototext, const string &caffemodel,
                    const int batch_size)
      : batch_size(batch_size) {
    net = caffe_make_net_from_file_content(prototext, caffemodel, batch_size);
    _DEBUG_PRINT(batch_size, net->blob_by_name("data")->shape(),
                 net->blob_by_name("prob")->shape());
  }
  std::unique_ptr<caffe::Net<float>> net;
  int batch_size;
  vector<float> forward(const vector<float> &cls_input_batch) {
    input_coarse_shape.assert_shape_match(cls_input_batch, batch_size);
    auto blob_data = net->blob_by_name("data")->data();
    CHECK_EQ(blob_data->size(), input_coarse_shape.single_batch_size() *
                                    sizeof(float) * batch_size);
    memcpy(blob_data->mutable_cpu_data(), cls_input_batch.data(),
           blob_data->size());
    net->Forward();
    vector<float> output_coarse(output_coarse_shape.single_batch_size() *
                                batch_size);
    auto prob_data = net->blob_by_name("prob")->data();
    memcpy(output_coarse.data(), prob_data->cpu_data(), prob_data->size());
    return output_coarse;
  }
};
}  // namespace terror_mixup
}  // namespace tron
#endif
