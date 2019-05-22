#pragma once

#include <string>
#include <vector>

namespace tensord {
namespace core {

template <typename DType>
class NetIn {
 public:
  NetIn() = default;
  NetIn(std::vector<std::string> names,
        std::vector<std::vector<DType>> datas)
      : names(names), datas(datas) {}
  std::vector<std::string> names;
  std::vector<std::vector<DType>> datas;

  std::vector<DType> &GetByName(std::string name) {
    for (std::size_t i = 0; i < names.size(); i++) {
      if (names[i] == name) {
        return datas[i];
      }
    }
    return none;
  }

 private:
  std::vector<DType> none;
};

template <typename DType>
class NetOut {
 public:
  NetOut() = default;
  NetOut(std::vector<std::string> names,
         std::vector<std::vector<DType>> datas)
      : names(names), datas(datas) {}
  std::vector<std::string> names;
  std::vector<std::vector<DType>> datas;

  std::vector<DType> &GetByName(std::string name) {
    for (std::size_t i = 0; i < names.size(); i++) {
      if (names[i] == name) {
        return datas[i];
      }
    }
    return none;
  }

 private:
  std::vector<DType> none;
};

template <typename DType>
class Net {
 public:
  virtual ~Net() = 0;
  virtual void Init() = 0;
  virtual void Predict(const std::vector<NetIn<DType>> &,
                       std::vector<NetOut<DType>> *) = 0;
  virtual void Release() = 0;
};

template <typename DType>
inline Net<DType>::~Net() {}

}  // namespace core
}  // namespace tensord
