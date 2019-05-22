#ifndef TRON_FRAMEWORK_FORWARD_HPP  // NOLINT
#define TRON_FRAMEWORK_FORWARD_HPP

#include <algorithm>
#include <future>  // NOLINT
#include <map>
#include <memory>
#include <string>
#include <thread>  // NOLINT
#include <utility>
#include <vector>

#include <glog/logging.h>  // NOLINT
#include <zmq.hpp>
#include <zmq_addon.hpp>

#include "common/type.hpp"
#include "pp.hpp"

namespace tron {
namespace framework {

template <class Forward, class Request, class Response>
class ForwardWrap {
 public:
  ForwardWrap() = default;
  ~ForwardWrap() { release(); }

  void Setup(const VecInt &);
  void AddContainer(std::shared_ptr<Forward> forward);
  void SetupBackend(std::shared_ptr<zmq::context_t> context,
                    const std::string name,
                    const std::string &addr_frontend,
                    const std::string &addr_backend);
  void SetupFrontend(std::shared_ptr<zmq::context_t> context,
                     const std::string &addr_frontend,
                     const std::string &identity);
  void Predict(const std::vector<Request> &requests,
               std::vector<Response> *responses);

 private:
  void release();

  enum _Mode_ {
    Standalone,
    Frontend,
    Backend
  };
  _Mode_ mode_;

  VecInt in_shape_;
  std::vector<std::shared_ptr<Forward>> nets_;

  std::shared_ptr<zmq::socket_t> client_;

  std::vector<std::promise<void>> worker_promises_;
  std::vector<std::thread> worker_threads_;

  std::shared_ptr<std::promise<void>> queue_promise_;
  std::shared_ptr<std::thread> queue_thread_;
};

template <class Forward, class Request, class Response>
void ForwardWrap<Forward, Request, Response>::Setup(const VecInt &in_shape) {
  in_shape_ = in_shape;
}

template <class Forward, class Request, class Response>
void ForwardWrap<Forward, Request, Response>::AddContainer(
    std::shared_ptr<Forward> forward) {
  nets_.push_back(forward);
}

template <class Forward, class Request, class Response>
void ForwardWrap<Forward, Request, Response>::SetupBackend(
    std::shared_ptr<zmq::context_t> context,
    const std::string name,
    const std::string &addr_frontend,
    const std::string &addr_backend) {
  CHECK_GE(nets_.size(), 1);

  for (std::size_t i = 0; i < nets_.size(); i++) {
    auto net = nets_[i];
    std::promise<void> e;
    std::future<void> future = e.get_future();
    auto func =
        [net](const std::vector<std::pair<size_t, const void *>> &requests,
              std::vector<std::vector<unsigned char>> *responses) {
          std::vector<Request> _requests;
          std::vector<Response> _responses;
          for (auto req0 : requests) {
            Request req;
            req.ParseFromArray(req0.second, req0.first);
            _requests.push_back(req);
            _responses.push_back(Response());
          }
          LOG(INFO) << "forward predict begin ....";
          net->Predict(_requests, &_responses);
          LOG(INFO) << "forward predict end. " << requests.size() << " "
                    << responses->size() << " " << _responses.size();
          for (std::size_t i = 0; i < responses->size(); i++) {
            auto &resp0 = responses->at(i);
            auto &resp = _responses[i];
            resp0.resize(resp.ByteSize());
            resp.SerializeToArray(&resp0[0], resp.ByteSize());
          }
          LOG(INFO) << "worker predict end.";
        };
    auto func0 =
        [net, context, addr_backend, func](std::future<void> exit) {
          net->Init();
          worker(std::move(exit), context, addr_backend, func);
        };
    worker_threads_.emplace_back(func0, std::move(future));
    // worker_threads_.emplace_back(worker, std::move(future),
    //                              context, addr_backend, func);
    worker_promises_.push_back(std::move(e));
  }

  queue_promise_ = std::make_shared<std::promise<void>>();
  std::future<void> future = queue_promise_->get_future();
  queue_thread_ = std::make_shared<std::thread>(
      batchQueue, std::move(future), context,
      name, in_shape_[0],
      addr_frontend, addr_backend);

  mode_ = Backend;
  return;
}
template <class Forward, class Request, class Response>
void ForwardWrap<Forward, Request, Response>::SetupFrontend(
    std::shared_ptr<zmq::context_t> context,
    const std::string &addr_frontend,
    const std::string &identity) {
  client_ = std::make_shared<zmq::socket_t>(*context, ZMQ_DEALER);
  client_->setsockopt(ZMQ_IDENTITY, identity);
  client_->connect(addr_frontend);
  mode_ = Frontend;
  return;
}

template <class Forward, class Request, class Response>
void ForwardWrap<Forward, Request, Response>::Predict(
    const std::vector<Request> &requests, std::vector<Response> *responses) {
  switch (mode_) {
    case Frontend: {
      for (auto req : requests) {
        auto body = req.SerializeAsString();
        zmq::multipart_t msg(body);
        msg.send(*client_);
      }

      for (auto cur = responses->begin(); cur != responses->end(); cur++) {
        zmq::multipart_t ret(*client_);
        cur->ParseFromString(ret.peekstr(ret.size() - 1));
      }
      break;
    }
    case Backend: {
      // NOTHING TO DO
      break;
    }
    case Standalone:
    default: {
      CHECK_GE(nets_.size(), 1);
      nets_[0]->Predict(requests, responses);
    }
  }
  return;
}

template <class Forward, class Request, class Response>
void ForwardWrap<Forward, Request, Response>::release() {
  for (std::size_t i = 0; i < worker_promises_.size(); i++) {
    worker_promises_[i].set_value();
  }
  for (std::size_t i = 0; i < worker_threads_.size(); i++) {
    if (worker_threads_[i].joinable()) worker_threads_[i].join();
  }

  if (queue_promise_) {
    queue_promise_->set_value();
  }
  if (queue_thread_) {
    queue_thread_->join();
  }
}

}  // namespace framework
}  // namespace tron

#endif  // TRON_FRAMEWORK_FORWARD_HPP NOLINT