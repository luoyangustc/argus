#ifndef TRON_FRAMEWORK_CONTEXT_HPP  // NOLINT
#define TRON_FRAMEWORK_CONTEXT_HPP

#include <memory>
#include <string>
#include <vector>

#include <zmq.hpp>
#include "glog/logging.h"

#include "base.hpp"
#include "common/archiver.hpp"
#include "common/protobuf.hpp"
#include "forward.hpp"
#include "proto/inference.pb.h"

namespace tron {
namespace framework {

template <class Params>
class Config {
 public:
  int batch_size;
  Params params;
};

template <class Params>
class Handle {
 public:
  ~Handle() { Release(); }

  template <class Forward, class Request, class Response>
  std::shared_ptr<ForwardWrap<Forward, Request, Response>> AddForward(
      void *ctx, const std::string &frontend);

  template <typename Inference>
  void Setup(void *ctx, std::shared_ptr<Inference> infer);

  template <typename Inference>
  std::shared_ptr<Inference> GetInference() {
    return std::static_pointer_cast<Inference>(inference_);
  }

  void Return(const inference::InferenceResponses &responses,
              void **out_data, int *out_size);

  void Release();

  Config<Params> config_;

  std::vector<std::shared_ptr<void>> frontends_ = {};
  std::shared_ptr<void> inference_;

  std::vector<char> out_data_;
};

template <class Params>
class Context {
 public:
  ~Context() { Release(); }

  int ParseConfig(const void *in_data, const int in_size,
                  const std::vector<std::vector<std::string>> &filenames,
                  std::vector<std::vector<std::vector<char>>> *files,
                  Config<Params> *config);

  template <class Forward, class Request, class Response, class Conf = Void>
  void AddForward(const std::vector<std::vector<char>> &net_param_data,
                  const std::vector<int> &in_shape, const int &gpu_id,
                  const std::string &frontend, const std::string &backend,
                  const std::string name, const int num = 1,
                  const Conf &config = Conf());

  void Release();

  Config<Params> config_;

  std::shared_ptr<zmq::context_t> zmq_context_;
  std::vector<std::shared_ptr<void>> backends_ = {};
  std::vector<Handle<Params> *> handles_;
};

template <class T>
static inline Context<T> *CreateContext() {
  google::InitGoogleLogging("QT");
  google::SetStderrLogging(google::INFO);
  google::InstallFailureSignalHandler();
  auto ctx = new Context<T>();
  ctx->zmq_context_ = std::make_shared<zmq::context_t>(1);
  return ctx;
}
template <class T>
static inline Handle<T> *CreateHandle() {
  auto handle = new Handle<T>();
  handle->out_data_ = std::vector<char>(1024 * 1024 * 4);
  return handle;
}

template <class T>
int Context<T>::ParseConfig(
    const void *in_data, const int in_size,
    const std::vector<std::vector<std::string>> &filenames,
    std::vector<std::vector<std::vector<char>>> *files,
    Config<T> *config) {
  LOG(INFO) << "create start";

  inference::CreateParams create_params;
  bool success = tron::read_proto_from_array(in_data, in_size, &create_params);
  if (!success) {
    LOG(ERROR) << "Parsing CreateParams Error! " << success;
    return 1;
  }
  Config<T> _config;
  _config.batch_size = create_params.batch_size();
  {
    JsonReader jReader(create_params.mutable_custom_params()->c_str());
    jReader &_config.params;
  }

  config_ = _config;

  for (std::size_t i = 0; i < filenames.size(); i++) {
    std::vector<std::vector<char>> _files(filenames[i].size());
    for (std::size_t j = 0; j < filenames[i].size(); j++) {
      const std::string &name = filenames[i][j];
      bool found = false;
      for (int k = 0; k < create_params.model_files_size(); k++) {
        auto &file = create_params.model_files(k);
        if (file.has_name() && file.name().find(name) != std::string::npos) {
          LOG(INFO) << file.name() << " " << file.body().size();
          std::vector<char> body(file.body().size());
          memcpy(&body[0], file.body().data(), file.body().size());
          _files[j] = body;
          found = true;
          break;
        }
      }
      if (found) continue;
      for (int k = 0; k < create_params.custom_files_size(); k++) {
        auto &file = create_params.custom_files(k);
        if (file.has_name() && file.name().find(name) != std::string::npos) {
          LOG(INFO) << file.name() << " " << file.body().size();
          std::vector<char> body(file.body().size());
          memcpy(&body[0], file.body().data(), file.body().size());
          _files[j] = body;
          found = true;
          break;
        }
      }
      if (found) continue;
      _files[j] = std::vector<char>();
    }
    files->push_back(_files);
  }

  *config = _config;
  return 0;
}

template <class T>
template <class Forward, class Request, class Response, class Conf>
void Context<T>::AddForward(
    const std::vector<std::vector<char>> &net_param_data,
    const std::vector<int> &in_shape, const int &gpu_id,
    const std::string &frontend, const std::string &backend,
    const std::string name, const int num,
    const Conf &config) {
  auto _backend = std::make_shared<ForwardWrap<Forward, Request, Response>>();
  _backend->Setup(in_shape);
  for (int i = 0; i < num; i++) {
    auto forward = std::make_shared<Forward>();
    forward->Setup(net_param_data, in_shape, gpu_id, config);
    _backend->AddContainer(forward);
  }
  _backend->SetupBackend(zmq_context_, name, frontend, backend);

  backends_.push_back(_backend);
}

template <class T>
void Context<T>::Release() {
  for (std::size_t i = 0; i < handles_.size(); i++) {
    delete handles_[i];
  }
  for (std::size_t i = 0; i < backends_.size(); i++) {
    backends_[i].reset();
  }
  // zmq_context_->close();
}

template <class T>
template <class Forward, class Request, class Response>
std::shared_ptr<ForwardWrap<Forward, Request, Response>> Handle<T>::AddForward(
    void *c, const std::string &frontend) {
  auto ctx = reinterpret_cast<Context<T> *>(c);
  auto _frontend = std::make_shared<ForwardWrap<Forward, Request, Response>>();
  _frontend->SetupFrontend(ctx->zmq_context_, frontend,
                           std::to_string(ctx->handles_.size()));
  frontends_.push_back(_frontend);
  return _frontend;
}

template <class T>
template <typename Inference>
void Handle<T>::Setup(void *c, std::shared_ptr<Inference> infer) {
  auto ctx = reinterpret_cast<Context<T> *>(c);
  config_ = ctx->config_;
  inference_ = infer;
  ctx->handles_.push_back(this);
}

template <class T>
void Handle<T>::Return(const inference::InferenceResponses &responses,
                       void **out_data, int *out_size) {
  // Return 会把数据写到成员变量out_data_
  // 返回参数out_data这个指针会指向成员变量out_data_
  // 其它地方不需要释放这块内存
  auto size = responses.ByteSize();
  CHECK_GE(out_data_.size(), size);
  responses.SerializeToArray(&out_data_[0], responses.ByteSize());
  *out_data = &out_data_[0];
  *out_size = responses.ByteSize();
}

template <class T>
void Handle<T>::Release() {}

}  // namespace framework
}  // namespace tron

#endif  // TRON_FRAMEWORK_CONTEXT_HPP NOLINT