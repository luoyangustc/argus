#ifndef SHADOW_CORE_NETWORK_HPP
#define SHADOW_CORE_NETWORK_HPP

#include "operator.hpp"
#include "workspace.hpp"

namespace Shadow {

class Network {
 public:
  void Setup(int device_id = 0);

  void LoadModel(const std::string &proto_bin);
  void LoadModel(const tron::NetParam &net_param);
  void LoadModel(const std::string &proto_str,
                 const std::vector<const void *> &weights);
  void LoadModel(const std::string &proto_str, const float *weights_data);

  void Reshape(const std::map<std::string, std::vector<int>> &shape_map = {});
  void Forward(const std::map<std::string, float *> &data_map = {});
  void Release();

  const Operator *GetOpByName(const std::string &op_name) {
    for (const auto &op : ops_) {
      if (op_name == op->name()) return op;
    }
    return nullptr;
  }
  template <typename T>
  const Blob<T> *GetBlobByName(const std::string &blob_name) const {
    return ws_.GetBlob<T>(blob_name);
  }
  template <typename T>
  Blob<T> *GetBlobByName(const std::string &blob_name) {
    return ws_.GetBlob<T>(blob_name);
  }
  template <typename T>
  const T *GetBlobDataByName(const std::string &blob_name) {
    auto *blob = ws_.GetBlob<T>(blob_name);
    if (blob == nullptr) {
      LOG(FATAL) << "Unknown blob: " + blob_name;
    } else {
      return blob->cpu_data();
    }
    return nullptr;
  }

  const std::vector<int> num_class() {
    VecInt num_classes;
    for (const auto dim : net_param_.num_class()) {
      num_classes.push_back(dim);
    }
    return num_classes;
  }
  const std::vector<std::string> out_blob() {
    VecString out_blobs;
    for (const auto &blob : net_param_.out_blob()) {
      out_blobs.push_back(blob);
    }
    return out_blobs;
  }
  const std::vector<std::string> in_blob() {
    VecString in_blobs;
    for (const auto &blob : net_param_.op(0).top()) {
      in_blobs.push_back(blob);
    }
    return in_blobs;
  }

  bool has_argument(const std::string &name) const {
    return arg_helper_.HasArgument(name);
  }
  template <typename T>
  T get_single_argument(const std::string &name, const T &default_value) const {
    return arg_helper_.template GetSingleArgument<T>(name, default_value);
  }
  template <typename T>
  bool has_single_argument_of_type(const std::string &name) const {
    return arg_helper_.template HasSingleArgumentOfType<T>(name);
  }
  template <typename T>
  const std::vector<T> get_repeated_argument(
      const std::string &name, const std::vector<T> &default_value = {}) const {
    return arg_helper_.template GetRepeatedArgument<T>(name, default_value);
  }

 private:
  void LoadProtoBin(const std::string &proto_bin, tron::NetParam *net_param);
  void LoadProtoStrOrText(const std::string &proto_str_or_text,
                          tron::NetParam *net_param);

  void Initial();

  void CopyWeights(const std::vector<const void *> &weights);
  void CopyWeights(const float *weights_data);

  tron::NetParam net_param_;
  ArgumentHelper arg_helper_;

  VecOp ops_;
  Workspace ws_;
};

}  // namespace Shadow

#endif  // SHADOW_CORE_NETWORK_HPP
