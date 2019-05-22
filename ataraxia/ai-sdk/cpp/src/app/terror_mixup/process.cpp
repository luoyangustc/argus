#include "process.hpp"
#include <functional>
#include <iostream>
#include <map>
#include <numeric>
#include <string>
#include <vector>
#include "glog/logging.h"
#include "google/protobuf/text_format.h"
#include "gsl/span"

namespace tron {
namespace terror_mixup {
using gsl::make_span;
using gsl::span;
using std::string;
using std::vector;

std::unique_ptr<caffe::Net<float>> caffe_make_net(
    const caffe::NetParameter &net_param,
    const caffe::NetParameter &model_param) {
  const auto glog_old_level = FLAGS_stderrthreshold;
  auto _ = gsl::finally([&] { google::SetStderrLogging(glog_old_level); });
  google::SetStderrLogging(google::ERROR);
  auto net = std::make_unique<caffe::Net<float>>(net_param);
  net->CopyTrainedLayersFrom(model_param);
  return net;
}

std::unique_ptr<caffe::Net<float>> caffe_make_net_from_file_content(
    const string &prototext, const string &caffemodel, const int batch_size) {
  caffe::NetParameter net_param, model_param;

  bool ok =
      google::protobuf::TextFormat::ParseFromString(prototext, &net_param);
  CHECK_EQ(ok, true);

  ok = model_param.ParsePartialFromString(caffemodel);
  CHECK_EQ(ok, true);

  net_param.mutable_layer(0)->mutable_input_param()->mutable_shape(0)->set_dim(
      0, batch_size);

  net_param.mutable_state()->set_phase(caffe::Phase::TEST);
  model_param.mutable_state()->set_phase(caffe::Phase::TEST);

  auto net = caffe_make_net(net_param, model_param);
  return net;
}

}  // namespace terror_mixup
}  // namespace tron
