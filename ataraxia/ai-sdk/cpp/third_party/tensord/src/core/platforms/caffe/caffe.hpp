#pragma once

#include <map>
#include <string>
#include <vector>

#include "caffe/caffe.hpp"

#include "core/platforms/net_base.hpp"

namespace tensord {
namespace core {
namespace platform {

class Caffe final : public NetBase<float> {
 public:
  static std::shared_ptr<Net<float>> Create(proto::Model,
                                            proto::Instance::Kind,
                                            int,
                                            int);

  ~Caffe() override {}
  void Init() override;
  void Predict(const std::vector<NetIn<float>> &,
               std::vector<NetOut<float>> *) override;
  void Release() override;

 private:
  std::shared_ptr<caffe::Net<float>> net_;
};

}  // namespace platform
}  // namespace core
}  // namespace tensord
