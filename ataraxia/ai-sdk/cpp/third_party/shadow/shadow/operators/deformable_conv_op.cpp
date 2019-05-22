#include "deformable_conv_op.hpp"

#include "activate_op.hpp"

namespace Shadow {

void DeformableConvOp::Reshape() {
  const auto *bottom = bottoms<float>(0);
  auto *top = mutable_tops<float>(0);

  CHECK_NE(bottom, top);

  int in_c = bottom->shape(1), in_h = bottom->shape(2), in_w = bottom->shape(3);

  VecInt top_shape = bottom->shape();
  top_shape[1] = num_output_;
  top_shape[2] =
      deformable_conv_out_size(in_h, kernel_size_, stride_, pad_, dilation_);
  top_shape[3] =
      deformable_conv_out_size(in_w, kernel_size_, stride_, pad_, dilation_);
  top->reshape(top_shape);

  out_spatial_dim_ = top->count(2);
  kernel_dim_ = kernel_size_ * kernel_size_ * in_c / group_;

  weight_offset_ = num_output_ * kernel_dim_ / group_;
  col_offset_ = kernel_dim_ * out_spatial_dim_;
  output_offset_ = num_output_ * out_spatial_dim_ / group_;

  if (bias_term_) {
    biases_multiplier_ =
        op_ws_->CreateBlob<float>(op_name_ + "_biases_multiplier");
    biases_multiplier_->reshape({out_spatial_dim_});
    Blas::Set(out_spatial_dim_, 1, biases_multiplier_->mutable_data(), 0);
  }
  col_image_ = op_ws_->CreateBlob<float>(op_name_ + "_col_image");
  col_image_->reshape({kernel_dim_ * group_, out_spatial_dim_});

  DLOG(INFO) << op_name_ << "(" << op_type_ << "): " << bottom->name()
             << Util::format_vector(bottom->shape(), ",", "(", ")") << " -> "
             << num_output_ << "_" << kernel_size_ << "x" << kernel_size_
             << "_s" << stride_ << "_p" << pad_ << " -> " << top->name()
             << Util::format_vector(top->shape(), ",", "(", ")");
}

void DeformableConvOp::Forward() {
  const auto *bottom = bottoms<float>(0);
  const auto *offset_blob = bottoms<float>(1);
  auto *top = mutable_tops<float>(0);

  int batch = bottom->shape(0);
  int top_num = top->num(), bottom_num = bottom->num();

  for (int b = 0; b < batch; ++b) {
    Vision::DeformableIm2Col(
        bottom->data(), bottom->shape(), offset_blob->data(), b * bottom_num,
        deformable_group_, kernel_size_, stride_, pad_, dilation_, 0,
        top->shape(), col_image_->mutable_data());
    for (int g = 0; g < group_; ++g) {
      Blas::BlasSgemm(0, 0, num_output_ / group_, out_spatial_dim_, kernel_dim_,
                      1, blobs<float>(0)->data(), weight_offset_ * g,
                      col_image_->data(), col_offset_ * g, 0,
                      top->mutable_data(), b * top_num + output_offset_ * g);
    }
    if (bias_term_) {
      Blas::BlasSgemm(0, 0, num_output_, out_spatial_dim_, 1, 1,
                      blobs<float>(1)->data(), 0, biases_multiplier_->data(), 0,
                      1, top->mutable_data(), b * top_num);
    }
  }
  if (activate_type_ == 1) {
    Vision::Activate(top->mutable_data(), top->count(), activate_type_);
  }
}

REGISTER_OPERATOR(DeformableConv, DeformableConvOp);

namespace Vision {

#if !defined(USE_CUDA) & !defined(USE_CL)
inline float deformable_im2col_bilinear(const float *bottom_data,
                                        int data_width, int height, int width,
                                        float h, float w) {
  auto h_low = static_cast<int>(std::floor(h));
  auto w_low = static_cast<int>(std::floor(w));
  int h_high;
  int w_high;
  if (h_low >= height - 1) {
    h_high = h_low = height - 1;
    h = (float)h_low;
  } else {
    h_high = h_low + 1;
  }
  if (w_low >= width - 1) {
    w_high = w_low = width - 1;
    w = (float)w_low;
  } else {
    w_high = w_low + 1;
  }
  float lh = h - h_low;
  float lw = w - w_low;
  float hh = 1 - lh, hw = 1 - lw;
  float v1 = bottom_data[h_low * data_width + w_low];
  float v2 = bottom_data[h_low * data_width + w_high];
  float v3 = bottom_data[h_high * data_width + w_low];
  float v4 = bottom_data[h_high * data_width + w_high];
  float w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;
  float val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  return val;
}

template <typename T>
void DeformableIm2Col(const T *in_data, const VecInt &in_shape,
                      const T *offset_data, int offset, int deformable_group,
                      int kernel_size, int stride, int pad, int dilation,
                      int zero_point, const VecInt &out_shape, T *out_data) {
  int in_c = in_shape[1], in_h = in_shape[2], in_w = in_shape[3];
  int out_h = out_shape[2], out_w = out_shape[3];
  int channel_per_deformable_group = in_c / deformable_group;
  for (int c_im = 0; c_im < in_c; ++c_im) {
    for (int h_col = 0; h_col < out_h; ++h_col) {
      for (int w_col = 0; w_col < out_w; ++w_col) {
        int c_col = c_im * kernel_size * kernel_size;
        int deformable_group_index = c_im / channel_per_deformable_group;
        int h_in = h_col * stride - pad;
        int w_in = w_col * stride - pad;
        T *data_col_ptr = out_data + (c_col * out_h + h_col) * out_w + w_col;
        const T *data_im_ptr =
            in_data + offset + (c_im * in_h + h_in) * in_w + w_in;
        const T *data_offset_ptr = offset_data + deformable_group_index * 2 *
                                                     kernel_size * kernel_size *
                                                     out_h * out_w;
        for (int i = 0; i < kernel_size; ++i) {
          for (int j = 0; j < kernel_size; ++j) {
            int data_offset_h_ptr =
                ((2 * (i * kernel_size + j)) * out_h + h_col) * out_w + w_col;
            int data_offset_w_ptr =
                ((2 * (i * kernel_size + j) + 1) * out_h + h_col) * out_w +
                w_col;
            T offset_h = data_offset_ptr[data_offset_h_ptr];
            T offset_w = data_offset_ptr[data_offset_w_ptr];
            auto val = static_cast<T>(zero_point);
            T h_im = h_in + i * dilation + offset_h;
            T w_im = w_in + j * dilation + offset_w;
            if (h_im >= 0 && w_im >= 0 && h_im < in_h && w_im < in_w) {
              T map_h = i * dilation + offset_h;
              T map_w = j * dilation + offset_w;
              int cur_height = in_h - h_in;
              int cur_width = in_w - w_in;
              val = deformable_im2col_bilinear(data_im_ptr, in_w, cur_height,
                                               cur_width, map_h, map_w);
            }
            *data_col_ptr = val;
            data_col_ptr += out_h * out_w;
          }
        }
      }
    }
  }
}

template void DeformableIm2Col(const float *in_data, const VecInt &in_shape,
                               const float *offset_data, int offset,
                               int deformable_group, int kernel_size,
                               int stride, int pad, int dilation,
                               int zero_point, const VecInt &out_shape,
                               float *out_data);

#elif defined(USE_CL)
#endif

}  // namespace Vision

}  // namespace Shadow
