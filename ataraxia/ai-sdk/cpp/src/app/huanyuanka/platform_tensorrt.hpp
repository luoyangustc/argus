#pragma once

#include <map>
#include <string>
#include <vector>

#include <cuda_runtime_api.h>

#include "MixPluginFactory.hpp"
#include "tensord/core/utils.hpp"
#include "tensord/tensord.hpp"

namespace tron {
namespace mix {

class TensorRT : public tensord::core::Net<float> {
 public:
  static std::shared_ptr<Net<float>> Create(tensord::proto::Model,
                                            tensord::proto::Instance::Kind,
                                            int,
                                            int);

  ~TensorRT() override {}
  void Init() override;
  void Predict(const std::vector<tensord::core::NetIn<float>> &,
               std::vector<tensord::core::NetOut<float>> *) override;
  void Release() override;

 protected:
  MixPluginFactory pluginFactory_;

  cudaStream_t stream_;
  nvinfer1::IRuntime *rt_;
  nvinfer1::ICudaEngine *engine_;
  nvinfer1::IExecutionContext *context_;
  std::map<std::string, void *> input_gpu_;
  std::map<std::string, void *> output_gpu_;

  tensord::proto::Model model_;
  tensord::proto::Instance::Kind kind_;
  int gpu_id_;
  int batch_size_;

  tensord::core::Buf<float> input_buf_;
  std::map<std::string, int> input_shape_;
  tensord::core::Buf<float> output_buf_;
  std::map<std::string, int> output_shape_;
};

}  // namespace mix
}  // namespace tron
