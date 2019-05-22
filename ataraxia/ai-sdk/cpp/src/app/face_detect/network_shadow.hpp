#ifndef TRON_NETWORK_SHADOW_HPP
#define TRON_NETWORK_SHADOW_HPP

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "proto/tron.pb.h"

#include "common/type.hpp"

namespace tron {
namespace fd {

class ShadowNetwork {
 public:
  ShadowNetwork() = default;
  ~ShadowNetwork() { Release(); }

  void Setup(const tron::NetParam &net_param, int device_id = 0);
  void Reshape(const std::map<std::string, std::vector<int>> &shape_map = {});
  void Forward(const std::map<std::string, float *> &data_map = {});
  void Release();

  const VecInt &GetBlobShapeByName(const std::string &blob_name) const;
  int GetBlobShapeByName(const std::string &blob_name, int index) const;
  const float *GetBlobDataByName(const std::string &blob_name) const;

  const std::vector<std::string> out_blob();
  const std::vector<std::string> in_blob();

  bool has_argument(const std::string &name);
  float get_single_argument(const std::string &name,
                            const float &default_value) const;
  bool has_single_argument_of_type(const std::string &name) const;
  const std::vector<float> get_repeated_argument(
      const std::string &name,
      const std::vector<float> &default_value = {}) const;

 private:
  std::shared_ptr<void> net_;
};

}  // namespace fd
}  // namespace tron

#endif  // TRON_NETWORK_SHADOW_HPP NOLINT