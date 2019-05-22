#pragma once

#include <future>  // NOLINT

#include "concurrentqueue/blockingconcurrentqueue.h"
#include "tensord/core/net.hpp"

namespace tensord {
namespace core {

template <typename DType>
class Task {
 public:
  Task() = default;
  Task(const Task<DType> &&task)
      : in_(task.in_),
        promise_(std::move(task.promise_)),
        future_(std::move(task.future_)) {}
  Task(const Task<DType> &task) = delete;
  explicit Task(const NetIn<DType> &in) : in_(in) {
    future_ = promise_.get_future();
  }
  ~Task() {}
  void SetOut(const NetOut<DType> &out) { promise_.set_value(out); }
  void GetOut(NetOut<DType> *out) { *out = future_.get(); }
  const NetIn<DType> &GetIn() { return in_; }

 private:
  NetIn<DType> in_;
  std::promise<NetOut<DType>> promise_;
  std::future<NetOut<DType>> future_;
};

template <typename DType>
class Tasks {
  using pTask = std::shared_ptr<Task<DType>>;

 public:
  pTask Submit(const NetIn<DType> &in) {
    pTask task = std::make_shared<Task<DType>>(in);
    queue_.enqueue(task);
    return task;
  }
  bool TryWait(pTask &task) { return queue_.try_dequeue(task); }  // NOLINT
  template <typename Rep, typename Period>
  bool WaitTimed(pTask &task,  // NOLINT
                 std::chrono::duration<Rep, Period> const &timeout) {
    return queue_.wait_dequeue_timed(task, timeout);
  }

 private:
  moodycamel::BlockingConcurrentQueue<pTask> queue_;
};

}  // namespace core
}  // namespace tensord
