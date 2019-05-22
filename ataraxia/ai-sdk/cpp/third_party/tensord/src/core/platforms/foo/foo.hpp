#pragma once

#include <vector>

#include "glog/logging.h"

#include "tensord/core/net.hpp"
#include "tensord/proto/tensord.pb.h"

namespace tensord {
namespace core {
namespace platform {

class Foo : public Net<float> {
 public:
  static std::shared_ptr<Net<float>> Create(proto::Model model,
                                            proto::Instance::Kind,
                                            int,
                                            int) {
    for (int i = 0; i < model.file_size(); i++) {
      LOG(INFO) << model.file(i).body().size();
    }
    auto ptr = std::make_shared<Foo>();
    return std::static_pointer_cast<Net<float>>(ptr);
  }

  ~Foo() override {}
  void Init() override {}
  void Predict(const std::vector<NetIn<float>> &ins,
               std::vector<NetOut<float>> *outs) override {
    for (auto in = ins.begin(); in != ins.end(); in++) {
      NetOut<float> out;
      out.names = in->names;
      out.datas = in->datas;
      outs->push_back(out);
    }
  }
  void Release() override {}
};

}  // namespace platform
}  // namespace core
}  // namespace tensord
