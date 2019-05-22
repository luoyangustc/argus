#ifndef TRON_FOO_FORWARD_HPP  // NOLINT
#define TRON_FOO_FORWARD_HPP

#include <string>
#include <vector>

#include "common/type.hpp"
#include "forward.pb.h"  // NOLINT
#include "framework/base.hpp"

namespace tron {
namespace foo {

class Forward
    : public tron::framework::ForwardBase<inference::foo::ForwardRequest,
                                          inference::foo::ForwardResponse> {
  using Request = inference::foo::ForwardRequest;
  using Response = inference::foo::ForwardResponse;
  using Base = tron::framework::ForwardBase<Request, Response>;
  using Void = tron::framework::Void;

 public:
  Forward() = default;
  ~Forward() {}

  void Setup(const std::vector<std::vector<char>> &net_param_data,
             const VecInt &in_shape, const int &gpu_id = -1,
             const tron::framework::Void & = tron::framework::Void()) override;
  void Release() override;

 private:
  void Process(std::vector<Request>::const_iterator,
               std::vector<Request>::const_iterator,
               std::vector<Response>::iterator) override;
};

}  // namespace foo
}  // namespace tron

#endif  // TRON_FACE_FEATURE_FORWARD_FEATURE_HPP NOLINT