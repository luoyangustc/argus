#pragma once

#include <map>
#include <string>
#include <vector>

#include "mxnet/c_predict_api.h"

#include "core/platforms/net_base.hpp"

namespace tensord {
namespace core {
namespace platform {

class Mxnet final : public NetBase<float> {
 public:
  static std::shared_ptr<Net<float>> Create(proto::Model,
                                            proto::Instance::Kind,
                                            int,
                                            int);

  ~Mxnet() override {}
  void Init() override;
  void Predict(const std::vector<NetIn<float>> &,
               std::vector<NetOut<float>> *) override;
  void Release() override;

 private:
  PredictorHandle handle_;
};

}  // namespace platform
}  // namespace core
}  // namespace tensord
