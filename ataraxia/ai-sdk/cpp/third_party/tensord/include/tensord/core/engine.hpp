#pragma once

#include <vector>

#include "glog/logging.h"

#include "tensord/core/net.hpp"
#include "tensord/proto/tensord.pb.h"

namespace tensord {
namespace core {

template <typename DType>
class Engine {
 public:
  virtual ~Engine() = 0;
  virtual void Setup(std::function<
                         std::shared_ptr<Net<DType>>(
                             proto::Model,
                             proto::Instance::Kind,
                             int,
                             int)>,
                     const proto::Model &,
                     const proto::Instance &) = 0;
  virtual void Predict(const std::vector<NetIn<DType>> &,
                       std::vector<NetOut<DType>> *) = 0;
};

template <typename DType>
inline Engine<DType>::~Engine() {}

}  // namespace core
}  // namespace tensord
