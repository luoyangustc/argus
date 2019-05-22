#pragma once

#include <vector>

#include <glog/logging.h>  // NOLINT

#include "core/tasks.hpp"
#include "tensord/core/engine.hpp"
#include "tensord/core/net.hpp"
#include "tensord/proto/tensord.pb.h"

namespace tensord {
namespace core {

template <typename DType>
void CreateNets(
    std::function<
        std::shared_ptr<Net<DType>>(
            proto::Model,
            proto::Instance::Kind,
            int,
            int)>,
    const proto::Model &,
    const proto::Instance &,
    std::vector<std::shared_ptr<Net<DType>>> *nets,
    std::vector<int> *batch_sizes);

template <typename DType>
void RunNet(std::shared_ptr<Net<DType>> net,
            std::shared_ptr<Tasks<DType>> tasks,
            const int &batch_size);

template <typename DType>
class EngineImpl : public Engine<DType> {
 public:
  EngineImpl() = default;
  ~EngineImpl() override { Release(); }

  void Setup(std::function<
                 std::shared_ptr<Net<DType>>(
                     proto::Model,
                     proto::Instance::Kind,
                     int,
                     int)>
                 createNet,
             const proto::Model &modelConfig,
             const proto::Instance &instanceConfig) override;
  void Predict(const std::vector<NetIn<DType>> &,
               std::vector<NetOut<DType>> *) override;

 private:
  void Release();

  std::shared_ptr<Tasks<DType>> tasks_;
  std::vector<std::shared_ptr<Net<DType>>> nets_;
  std::vector<std::promise<void>> worker_promises_;
  std::vector<std::thread> worker_threads_;
};

template <typename DType>
void EngineImpl<DType>::Setup(std::function<
                                  std::shared_ptr<Net<DType>>(
                                      proto::Model,
                                      proto::Instance::Kind,
                                      int,
                                      int)>
                                  createNet,
                              const proto::Model &modelConfig,
                              const proto::Instance &instanceConfig) {
  tasks_ = std::make_shared<Tasks<DType>>();
  std::vector<int> batch_sizes;
  CreateNets<DType>(createNet,
                    modelConfig, instanceConfig,
                    &nets_, &batch_sizes);
  CHECK_EQ(batch_sizes.size(), nets_.size());
  for (std::size_t i = 0; i < nets_.size(); i++) {
    auto net = nets_[i];
    std::promise<void> end;
    auto func = [net](std::future<void> end,
                      std::shared_ptr<Tasks<DType>> tasks,
                      int batch_size) {
      net->Init();
      while (end.wait_for(std::chrono::microseconds(1)) ==
             std::future_status::timeout) {
        RunNet<DType>(net, tasks, batch_size);
      }
    };
    worker_threads_.emplace_back(func,
                                 end.get_future(),
                                 tasks_,
                                 batch_sizes[i]);
    worker_promises_.push_back(std::move(end));
  }
  return;
}

template <typename DType>
void EngineImpl<DType>::Predict(const std::vector<NetIn<DType>> &ins,
                                std::vector<NetOut<DType>> *outs) {
  std::vector<std::shared_ptr<Task<DType>>> tasks;
  for (auto in = ins.begin(); in != ins.end(); in++) {
    tasks.push_back(tasks_->Submit(*in));
  }
  for (auto task = tasks.begin(); task != tasks.end(); task++) {
    NetOut<DType> out;
    (*task)->GetOut(&out);
    outs->push_back(out);
  }
  return;
}

template <typename DType>
void EngineImpl<DType>::Release() {
  for (auto end = worker_promises_.begin();
       end != worker_promises_.end();
       end++) {
    end->set_value();
  }
  for (auto th = worker_threads_.begin();
       th != worker_threads_.end();
       th++) {
    th->join();
  }
  nets_.clear();
  return;
}

template <typename DType>
void CreateNets(
    std::function<
        std::shared_ptr<Net<DType>>(
            proto::Model,
            proto::Instance::Kind,
            int,
            int)>
        createNet,
    const proto::Model &modelConfig,
    const proto::Instance &instanceConfig,
    std::vector<std::shared_ptr<Net<DType>>> *nets,
    std::vector<int> *batch_sizes) {
  for (int i = 0; i < instanceConfig.count_size(); i++) {
    auto count = instanceConfig.count(i);
    for (int j = 0; j < count.count(); j++) {
      int batch_size = count.batchsize() > 0
                           ? count.batchsize()
                           : instanceConfig.batchsize();
      switch (count.kind()) {
        case proto::Instance_Kind_CPU: {
          nets->push_back(createNet(modelConfig, count.kind(), 0, batch_size));
          batch_sizes->push_back(batch_size);
          break;
        }
        case proto::Instance_Kind_GPU: {
          for (int k = 0; k < count.gpu_size(); k++) {
            auto gpu_id = count.gpu(k);
            nets->push_back(
                createNet(modelConfig, count.kind(), gpu_id, batch_size));
            batch_sizes->push_back(batch_size);
          }
          break;
        }
        default:
          break;
      }
    }
  }
  return;
}

template <typename DType>
void RunNet(std::shared_ptr<Net<DType>> net,
            std::shared_ptr<Tasks<DType>> tasks,
            const int &batch_size) {
  std::vector<std::shared_ptr<Task<DType>>> _tasks;
  std::vector<NetIn<DType>> ins;
  std::vector<NetOut<DType>> outs;

  for (int i = 0; i < batch_size; i++) {
    std::shared_ptr<Task<DType>> task;
    if (i == 0) {
      if (!tasks->WaitTimed(task, std::chrono::milliseconds(5))) {
        return;
      }
    } else {
      if (!tasks->TryWait(task)) {
        break;
      }
    }
    _tasks.push_back(task);
    ins.push_back(task->GetIn());
  }

  net->Predict(ins, &outs);
  CHECK_EQ(outs.size(), ins.size());
  for (std::size_t i = 0; i < outs.size(); i++) {
    _tasks[i]->SetOut(outs[i]);
  }
  return;
}

}  // namespace core
}  // namespace tensord
