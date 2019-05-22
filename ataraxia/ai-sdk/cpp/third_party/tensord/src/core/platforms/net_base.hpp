#pragma once

#include <map>
#include <string>
#include <vector>

#include "tensord/core/net.hpp"
#include "tensord/core/utils.hpp"
#include "tensord/proto/tensord.pb.h"

namespace tensord {
namespace core {
namespace platform {

template <typename DType>
class NetBase : public Net<DType> {
 protected:
  void setup(proto::Model model,
             proto::Instance::Kind kind,
             int gpu_id,
             int batch_size) {
    model_ = model;
    kind_ = kind;
    gpu_id_ = gpu_id;
    batch_size_ = batch_size;
  }
  void initBuf();

  proto::Model model_;
  proto::Instance::Kind kind_;
  int gpu_id_;
  int batch_size_;

  Buf<float> input_buf_;
  std::map<std::string, int> input_shape_;
  Buf<float> output_buf_;
  std::map<std::string, int> output_shape_;
};

template <typename DType>
void NetBase<DType>::initBuf() {
  {
    std::vector<std::string> input_keys;
    std::vector<int> input_shapes;
    for (int i = 0; i < model_.input_size(); i++) {
      input_keys.push_back(model_.input(i).name());
      int shape = 1;
      for (int j = 0; j < model_.input(i).shape_size(); j++) {
        shape *= model_.input(i).shape(j);
      }
      input_shapes.push_back(shape);
      input_shape_[model_.input(i).name()] = shape;
    }
    input_buf_ = Buf<DType>(batch_size_, input_keys, input_shapes);
  }
  {
    std::vector<std::string> output_keys;
    std::vector<int> output_shapes;
    for (int i = 0; i < model_.output_size(); i++) {
      output_keys.push_back(model_.output(i).name());
      int shape = 1;
      for (int j = 0; j < model_.output(i).shape_size(); j++) {
        shape *= model_.output(i).shape(j);
      }
      output_shapes.push_back(shape);
      output_shape_[model_.output(i).name()] = shape;
    }
    output_buf_ = Buf<DType>(batch_size_, output_keys, output_shapes);
  }
}

}  // namespace platform
}  // namespace core
}  // namespace tensord
