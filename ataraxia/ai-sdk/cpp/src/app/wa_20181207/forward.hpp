#ifndef TRON_WA_FORWARD_DET_HPP  // NOLINT
#define TRON_WA_FORWARD_DET_HPP

#include <string>
#include <utility>
#include <vector>

#include "common/type.hpp"
#include "forward.pb.h"  // NOLINT
#include "framework/base.hpp"
#include "net.hpp"

namespace tron {
namespace wa {

struct ForwardConfig {
  ForwardConfig() = default;
  ForwardConfig(int limit, float threshold, float nms_threshold)
      : limit(limit), threshold(threshold), nms_threshold(nms_threshold) {}
  ~ForwardConfig() {}

  int limit;
  float threshold;
  float nms_threshold;
};

class Forward
    : public framework::ForwardBase<inference::wa::ForwardRequest,
                                    inference::wa::ForwardResponse,
                                    ForwardConfig> {
  using ForwardRequest = inference::wa::ForwardRequest;
  using ForwardResponse = inference::wa::ForwardResponse;

 public:
  Forward() = default;
  ~Forward() noexcept { Release(); }

  void Setup(const std::vector<std::vector<char>> &net_param_data,
             const VecInt &in_shape, const int &gpu_id,
             const ForwardConfig &config) override;

  void Release() override;

 private:
  using Base = framework::ForwardBase<inference::wa::ForwardRequest,
                                      inference::wa::ForwardResponse,
                                      ForwardConfig>;
  void Process(std::vector<ForwardRequest>::const_iterator,
               std::vector<ForwardRequest>::const_iterator,
               std::vector<ForwardResponse>::iterator) override;

  Shadow::Net *net_;
  ForwardConfig config_;
};

}  // namespace wa
}  // namespace tron

#endif  // TRON_WA_FORWARD_DET_HPP NOLINT