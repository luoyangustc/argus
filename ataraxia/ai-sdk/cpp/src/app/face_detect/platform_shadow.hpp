#pragma once

#include <map>
#include <string>
#include <vector>

#include "network_shadow.hpp"
#include "tensord/tensord.hpp"

namespace tron {
namespace fd {

template <typename DType>
class Buf {
 public:
  Buf() = default;
  Buf(const int &, const std::vector<std::string> &, const std::vector<int> &);

  void Copy(const std::string &, const int &, const DType *);
  DType *Get(const std::string &, const int &);

  int batch_size_;
  // key: <index, shape * batch_size>
  std::map<std::string, std::pair<int, int>> keys_;
  std::vector<DType> buf_;
};

class Shadow : public tensord::core::Net<float> {
 public:
  static std::shared_ptr<Net<float>> Create(tensord::proto::Model,
                                            tensord::proto::Instance::Kind,
                                            int,
                                            int);

  ~Shadow() override {}
  void Init() override;
  void Predict(const std::vector<tensord::core::NetIn<float>> &,
               std::vector<tensord::core::NetOut<float>> *) override;
  void Release() override;

 protected:
  virtual void PostForward(const std::vector<tensord::core::NetIn<float>> &,
                           std::vector<tensord::core::NetOut<float>> *);

  ShadowNetwork net_;

  tensord::proto::Model model_;
  tensord::proto::Instance::Kind kind_;
  int gpu_id_;
  int batch_size_;

  Buf<float> input_buf_;
  std::map<std::string, int> input_shape_;
  Buf<float> output_buf_;
  std::map<std::string, int> output_shape_;
};

class ShadowFD : public Shadow {
 public:
  static std::shared_ptr<Net<float>> Create(tensord::proto::Model,
                                            tensord::proto::Instance::Kind,
                                            int,
                                            int);

  void Init() override;

 protected:
  void PostForward(const std::vector<tensord::core::NetIn<float>> &,
                   std::vector<tensord::core::NetOut<float>> *) override;
};

template <typename DType>
Buf<DType>::Buf(const int &batch_size,
                const std::vector<std::string> &keys,
                const std::vector<int> &shapes) {
  CHECK_GE(batch_size, 0);
  CHECK_EQ(keys.size(), shapes.size());

  batch_size_ = batch_size;

  int size = 0;
  for (std::size_t i = 0; i < keys.size(); i++) {
    keys_[keys[i]] = std::pair<int, int>(size, shapes[i]);
    size += batch_size * shapes[i];
  }
  buf_.resize(size);
}

template <typename DType>
void Buf<DType>::Copy(const std::string &key,
                      const int &index,
                      const DType *data) {
  CHECK_LE(index, batch_size_);
  memcpy(buf_.data() + keys_[key].first + index * keys_[key].second,
         data,
         keys_[key].second * sizeof(DType));
}

template <typename DType>
DType *Buf<DType>::Get(const std::string &key, const int &index) {
  CHECK_LE(index, batch_size_);
  return buf_.data() + keys_[key].first + index * keys_[key].second;
}

}  // namespace fd
}  // namespace tron
