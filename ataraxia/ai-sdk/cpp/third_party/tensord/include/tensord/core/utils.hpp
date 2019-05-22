#pragma once

#include <fstream>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "glog/logging.h"

namespace tensord {
namespace core {

inline std::vector<char> ReadFile(const std::string &filename) {
  std::ifstream in(filename, std::iostream::in | std::iostream::binary);
  if (!in) {
    throw std::invalid_argument("open file failed");
  }
  in.seekg(0, std::istream::end);
  int file_size = in.tellg();
  if (file_size == -1) {
    throw std::invalid_argument("seek file failed");
  }
  in.seekg(0, std::istream::beg);
  std::vector<char> result;
  result.resize(file_size);
  in.read(&result[0], file_size);
  return result;
}

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

}  // namespace core
}  // namespace tensord
