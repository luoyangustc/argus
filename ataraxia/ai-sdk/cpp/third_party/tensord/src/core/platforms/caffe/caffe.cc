#include "core/platforms/caffe/caffe.hpp"

#include "google/protobuf/text_format.h"

namespace tensord {
namespace core {
namespace platform {

std::shared_ptr<Net<float>> Caffe::Create(proto::Model model,
                                          proto::Instance::Kind kind,
                                          int gpu_id,
                                          int batch_size) {
  CHECK_EQ(kind, proto::Instance::GPU);
  auto net = std::make_shared<Caffe>();
  net->setup(model, kind, gpu_id, batch_size);
  return net;
}

void Caffe::Init() {
  std::string *net_prototxt;
  const char *net_caffemodel;
  int model_size;
  for (int i = 0; i < model_.file_size(); i++) {
    if (model_.mutable_file(i)->name() == "net.prototxt") {
      net_prototxt = model_.mutable_file(i)->mutable_body();
    } else if (model_.mutable_file(i)->name() == "net.caffemodel") {
      net_caffemodel = model_.mutable_file(i)->mutable_body()->data();
      model_size = model_.mutable_file(i)->mutable_body()->size();
    }
  }
  caffe::NetParameter net_param, model_param;
  google::protobuf::TextFormat::ParseFromString(*net_prototxt, &net_param);
  net_param.mutable_state()->set_phase(caffe::Phase::TEST);
  caffe::UpgradeNetAsNeeded("", &net_param);
  model_param.ParseFromArray(net_caffemodel, model_size);
  model_param.mutable_state()->set_phase(caffe::Phase::TEST);

  switch (kind_) {
    case proto::Instance::GPU:
      caffe::Caffe::set_mode(caffe::Caffe::GPU);
      caffe::Caffe::SetDevice(gpu_id_);
      break;
    case proto::Instance::CPU:
      caffe::Caffe::set_mode(caffe::Caffe::CPU);
      break;
    default:
      break;
  }

  net_ = std::make_shared<caffe::Net<float>>(net_param);
  net_->CopyTrainedLayersFrom(model_param);

  for (int i = 0; i < model_.input_size(); i++) {
    std::vector<int> shapes = {batch_size_};
    for (int j = 0; j < model_.mutable_input(i)->shape_size(); j++) {
      shapes.push_back(model_.mutable_input(i)->shape(j));
    }
    net_->blob_by_name(model_.mutable_input(i)->name())->Reshape(shapes);
  }
  net_->Reshape();

  NetBase<float>::initBuf();
}

void Caffe::Predict(const std::vector<NetIn<float>> &ins,
                    std::vector<NetOut<float>> *outs) {
  for (std::size_t i = 0; i < ins.size(); i++) {
    for (std::size_t j = 0; j < ins[i].names.size(); j++) {
      input_buf_.Copy(ins[i].names[j], i, ins[i].datas[j].data());
    }
  }
  for (const std::string &name : ins[0].names) {
    memcpy(net_->blob_by_name(name)->data()->mutable_cpu_data(),
           input_buf_.Get(name, 0),
           input_shape_[name] * batch_size_ * sizeof(float));
  }
  net_->Forward();
  for (int i = 0; i < model_.output_size(); i++) {
    memcpy(output_buf_.Get(model_.mutable_output(i)->name(), 0),
           net_->blob_by_name(model_.mutable_output(i)->name())->cpu_data(),
           output_shape_[model_.mutable_output(i)->name()] *
               batch_size_ * sizeof(float));
  }
  for (std::size_t i = 0; i < ins.size(); i++) {
    NetOut<float> out;
    for (auto iter = output_shape_.begin();
         iter != output_shape_.end();
         iter++) {
      out.names.push_back(iter->first);
      std::vector<float> _buf(iter->second);
      memcpy(_buf.data(),
             output_buf_.Get(iter->first, i),
             iter->second * sizeof(float));
      out.datas.emplace_back(_buf);
    }
    outs->push_back(out);
  }
}

void Caffe::Release() {}

}  // namespace platform
}  // namespace core
}  // namespace tensord
