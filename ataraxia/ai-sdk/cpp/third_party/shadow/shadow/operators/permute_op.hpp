#ifndef SHADOW_OPERATORS_PERMUTE_OP_HPP
#define SHADOW_OPERATORS_PERMUTE_OP_HPP

#include "core/operator.hpp"

namespace Shadow {

class PermuteOp : public Operator {
 public:
  explicit PermuteOp(const tron::OpParam &op_param, Workspace *ws)
      : Operator(op_param, ws) {
    permute_order_data_ = get_repeated_argument<int>("order");
    num_axes_ = static_cast<int>(permute_order_data_.size());
    CHECK_EQ(num_axes_, bottoms<float>(0)->num_axes());
  }

  void Reshape() override;
  void Forward() override;

 private:
  int num_axes_;
  VecInt permute_order_data_;

  BlobI *permute_order_ = nullptr, *old_steps_ = nullptr, *new_steps_ = nullptr;
};

namespace Vision {

template <typename T, typename Dtype>
void Permute(const T *in_data, int count, int num_axes,
             const Dtype *permute_order, const Dtype *old_steps,
             const Dtype *new_steps, T *out_data);

}  // namespace Vision

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_PERMUTE_OP_HPP
