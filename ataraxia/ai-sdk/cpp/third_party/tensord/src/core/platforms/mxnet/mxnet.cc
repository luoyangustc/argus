
#include "core/platforms/mxnet/mxnet.hpp"

#include <string>

#include "glog/logging.h"

namespace tensord {
namespace core {
namespace platform {

std::shared_ptr<Net<float>> Mxnet::Create(proto::Model model,
                                          proto::Instance::Kind kind,
                                          int gpu_id,
                                          int batch_size) {
  CHECK_EQ(kind, proto::Instance::GPU);
  auto net = std::make_shared<Mxnet>();
  net->setup(model, kind, gpu_id, batch_size);
  return net;
}

void Mxnet::Init() {
  const char *input_key[model_.input_size()];
  for (int i = 0; i < model_.input_size(); i++) {
    input_key[i] = model_.input(i).name().c_str();
  }
  const char **input_keys = input_key;
  const char *output_key[model_.output_size()];
  for (int i = 0; i < model_.output_size(); i++) {
    output_key[i] = model_.output(i).name().c_str();
  }
  const char **output_keys = output_key;

  mx_uint *input_shape_indptr = new mx_uint[model_.input_size() + 1];
  *(input_shape_indptr + 0) = 0;
  int n = 0;
  for (int i = 0; i < model_.input_size(); i++) {
    *(input_shape_indptr + 1 + i) = model_.input(i).shape_size() + 1;
    n += 1 + model_.input(i).shape_size();
  }
  mx_uint *input_shape_data = new mx_uint[n];
  n = 0;
  for (int i = 0; i < model_.input_size(); i++) {
    *(input_shape_data + n++) = batch_size_;
    for (int j = 0; j < model_.input(i).shape_size(); j++) {
      *(input_shape_data + n++) = model_.input(i).shape(j);
    }
  }

  const char *model_symbol_json;
  const char *model_params;
  int model_params_size;
  for (int i = 0; i < model_.file_size(); i++) {
    if (model_.mutable_file(i)->name() == "model-symbol.json") {
      model_symbol_json = model_.mutable_file(i)->mutable_body()->c_str();
    } else if (model_.mutable_file(i)->name() == "model.params") {
      model_params = model_.mutable_file(i)->mutable_body()->data();
      model_params_size = model_.mutable_file(i)->mutable_body()->size();
    }
  }
  MXPredCreatePartialOut(model_symbol_json,
                         model_params, model_params_size,
                         2, gpu_id_,
                         1, input_keys, input_shape_indptr, input_shape_data,
                         1, output_keys,
                         &handle_);

  delete[] input_shape_indptr;
  delete[] input_shape_data;

  if (handle_ == nullptr) {
    LOG(WARNING) << "create mxnet net failed. " << MXGetLastError();
    return;
  }

  NetBase<float>::initBuf();
}

void Mxnet::Predict(const std::vector<NetIn<float>> &ins,
                    std::vector<NetOut<float>> *outs) {
  for (std::size_t i = 0; i < ins.size(); i++) {
    for (std::size_t j = 0; j < ins[i].names.size(); j++) {
      input_buf_.Copy(ins[i].names[j], i, ins[i].datas[j].data());
    }
  }
  for (const std::string &name : ins[0].names) {
    MXPredSetInput(handle_,
                   name.c_str(),
                   input_buf_.Get(name, 0),
                   input_shape_[name] * batch_size_);
  }
  MXPredForward(handle_);
  for (int i = 0; i < model_.output_size(); i++) {
    MXPredGetOutput(handle_,
                    i,
                    output_buf_.Get(model_.output(i).name(), 0),
                    output_shape_[model_.output(i).name()] * batch_size_);
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

void Mxnet::Release() { MXPredFree(handle_); }

}  // namespace platform
}  // namespace core
}  // namespace tensord
