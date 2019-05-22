#include "forward.hpp"

#include "glog/logging.h"

#include "common/time.hpp"

namespace tron {
namespace foo {

void Forward::Setup(const std::vector<std::vector<char>> &,
                    const VecInt &in_shape, const int &,
                    const tron::framework::Void &) {
  Base::Setup(in_shape);
}

void Forward::Release() {}

void Forward::Process(
    std::vector<Request>::const_iterator requests_first,
    std::vector<Request>::const_iterator requests_last,
    std::vector<Response>::iterator responses_first) {
  int n = 0;
  for (auto cur = requests_first; cur != requests_last; ++cur) {
    auto body = const_cast<Request &>(*cur).mutable_data()->mutable_body();
    memcpy(&in_data_[in_num_ * n], &((*body)[0]), body->size());
    n++;
  }
  int size = n;

  for (int i = 0; i < size; ++i) {
    auto resp = responses_first + i;
    auto begin = in_data_.begin() + in_num_ * i;
    float sum = 0;
    for (auto cur = begin; cur != begin + in_num_; cur++) sum += *cur;
    resp->set_sum(sum);
  }
}

}  // namespace foo
}  // namespace tron
