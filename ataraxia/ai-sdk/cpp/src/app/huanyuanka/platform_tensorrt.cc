
#include "platform_tensorrt.hpp"

#include <string>

#include "glog/logging.h"

#include "Util.hpp"

namespace tron {
namespace mix {

static Shadow::Logger gLogger;
static Shadow::Profiler gProfiler;

std::shared_ptr<tensord::core::Net<float>> TensorRT::Create(
    tensord::proto::Model model,
    tensord::proto::Instance::Kind kind,
    int gpu_id,
    int batch_size) {
  auto net = std::make_shared<TensorRT>();
  net->model_ = model;
  net->kind_ = kind;
  net->gpu_id_ = gpu_id;
  net->batch_size_ = batch_size;
  return net;
}

void TensorRT::Init() {
  const char *model_params;
  int model_params_size;
  for (int i = 0; i < model_.file_size(); i++) {
    if (model_.mutable_file(i)->name() == "net.bin") {
      model_params = model_.mutable_file(i)->mutable_body()->data();
      model_params_size = model_.mutable_file(i)->mutable_body()->size();
    }
  }
  CHECK_GT(model_params_size, 0);

  cudaSetDevice(gpu_id_);
  rt_ = nvinfer1::createInferRuntime(gLogger);
  engine_ = rt_->deserializeCudaEngine(static_cast<const void *>(model_params),
                                       model_params_size,
                                       &pluginFactory_);
  context_ = engine_->createExecutionContext();
  context_->setProfiler(&gProfiler);
  // engine_->setMaxBatchSize(batch_size_);

  cudaStreamCreate(&stream_);

  int bindings = engine_->getNbBindings();
  for (int i = 0; i < bindings; i++) {
    auto dim = static_cast<DimsCHW &&>(engine_->getBindingDimensions(i));
    if (engine_->bindingIsInput(i)) {
      LOG(INFO) << i << " INPUT: "
                << dim.c() << " " << dim.w() << " " << dim.h();
    } else {
      LOG(INFO) << i << " OUTPUT: "
                << dim.c() << " " << dim.w() << " " << dim.h();
    }
  }

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
    input_buf_ = tensord::core::Buf<float>(batch_size_,
                                           input_keys,
                                           input_shapes);
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
    output_buf_ = tensord::core::Buf<float>(batch_size_,
                                            output_keys,
                                            output_shapes);
  }

  for (int i = 0; i < model_.input_size(); i++) {
    auto name = model_.input(i).name();
    void *ptr;
    cudaMalloc(&ptr, batch_size_ * input_shape_[name] * sizeof(float));
    input_gpu_[name] = ptr;
  }
  for (int i = 0; i < model_.output_size(); i++) {
    auto name = model_.output(i).name();
    void *ptr;
    cudaMalloc(&ptr, batch_size_ * output_shape_[name] * sizeof(float));
    output_gpu_[name] = ptr;
  }
}

void TensorRT::Predict(const std::vector<tensord::core::NetIn<float>> &ins,
                       std::vector<tensord::core::NetOut<float>> *outs) {
  for (std::size_t i = 0; i < ins.size(); i++) {
    for (std::size_t j = 0; j < ins[i].names.size(); j++) {
      input_buf_.Copy(ins[i].names[j], i, ins[i].datas[j].data());
    }
  }

  for (int i = 0; i < model_.input_size(); i++) {
    auto name = model_.input(i).name();
    cudaMemcpyAsync(input_gpu_[name],
                    input_buf_.Get(name, 0),
                    input_shape_[name] * ins.size() * sizeof(float),
                    cudaMemcpyHostToDevice);
  }

  void **buffers = static_cast<void **>(
      malloc(sizeof(void *) * (input_gpu_.size() + output_gpu_.size())));
  for (int i = 0; i < model_.input_size(); i++) {
    buffers[i] = input_gpu_[model_.input(i).name()];
  }
  for (int i = 0; i < model_.output_size(); i++) {
    buffers[i + model_.input_size()] = output_gpu_[model_.output(i).name()];
  }
  context_->enqueue(ins.size(), buffers, stream_, nullptr);
  free(buffers);
  for (int i = 0; i < model_.output_size(); i++) {
    auto name = model_.output(i).name();
    cudaMemcpyAsync(output_buf_.Get(name, 0),
                    output_gpu_[name],
                    output_shape_[name] * batch_size_ * sizeof(float),
                    cudaMemcpyDeviceToHost);
  }
  cudaStreamSynchronize(stream_);

  for (std::size_t i = 0; i < ins.size(); i++) {
    tensord::core::NetOut<float> out;
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

void TensorRT::Release() {
  for (auto it = input_gpu_.begin(); it != input_gpu_.end(); it++) {
    cudaFree(it->second);
  }
  for (auto it = output_gpu_.begin(); it != output_gpu_.end(); it++) {
    cudaFree(it->second);
  }
  context_->destroy();
  engine_->destroy();
  rt_->destroy();
  pluginFactory_.destroyPlugin();
}

}  // namespace mix
}  // namespace tron
