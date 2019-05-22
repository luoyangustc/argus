
#include "network_shadow.hpp"

#include "core/network.hpp"

namespace tron {
namespace fd {

void ShadowNetwork::Setup(const tron::NetParam &net_param,
                          int device_id) {
  std::shared_ptr<Shadow::Network> net = std::make_shared<Shadow::Network>();
  net->Setup(device_id);
  net->LoadModel(net_param);
  net_ = net;
}

void ShadowNetwork::Reshape(
    const std::map<std::string, std::vector<int>> &shape_map) {
  std::static_pointer_cast<Shadow::Network>(net_)->Reshape(shape_map);
}
void ShadowNetwork::Forward(const std::map<std::string, float *> &data_map) {
  std::static_pointer_cast<Shadow::Network>(net_)->Forward(data_map);
}
void ShadowNetwork::Release() {
  if (net_) {
    std::static_pointer_cast<Shadow::Network>(net_)->Release();
    net_ = nullptr;
  }
}

const VecInt &ShadowNetwork::GetBlobShapeByName(
    const std::string &blob_name) const {
  return std::static_pointer_cast<Shadow::Network>(net_)
      ->GetBlobByName<float>(blob_name)
      ->shape();
}
int ShadowNetwork::GetBlobShapeByName(
    const std::string &blob_name, int index) const {
  return std::static_pointer_cast<Shadow::Network>(net_)
      ->GetBlobByName<float>(blob_name)
      ->shape(index);
}
const float *ShadowNetwork::GetBlobDataByName(
    const std::string &blob_name) const {
  return std::static_pointer_cast<Shadow::Network>(net_)
      ->GetBlobDataByName<float>(blob_name);
}

const std::vector<std::string> ShadowNetwork::out_blob() {
  return std::static_pointer_cast<Shadow::Network>(net_)->out_blob();
}
const std::vector<std::string> ShadowNetwork::in_blob() {
  return std::static_pointer_cast<Shadow::Network>(net_)->in_blob();
}

bool ShadowNetwork::has_argument(const std::string &name) {
  return std::static_pointer_cast<Shadow::Network>(net_)->has_argument(name);
}
float ShadowNetwork::get_single_argument(
    const std::string &name, const float &default_value) const {
  return std::static_pointer_cast<Shadow::Network>(net_)
      ->get_single_argument<float>(name, default_value);
}
bool ShadowNetwork::has_single_argument_of_type(const std::string &name) const {
  return std::static_pointer_cast<Shadow::Network>(net_)
      ->has_single_argument_of_type<float>(name);
}
const std::vector<float> ShadowNetwork::get_repeated_argument(
    const std::string &name, const std::vector<float> &default_value) const {
  return std::static_pointer_cast<Shadow::Network>(net_)
      ->get_repeated_argument<float>(name, default_value);
}

}  // namespace fd
}  // namespace tron
