/*!
 * Copyright (c) 2015 by Contributors
 * \file roi_align.cc
 * \brief roi align operator
 * \author Ross Girshick, Kye-Hyeon Kim, Jian Guo
*/
#include "./roi_align-inl.h"
#include <mshadow/base.h>
#include <mshadow/tensor.h>
#include <mshadow/packet-inl.h>
#include <mshadow/dot_engine-inl.h>
#include <cassert>

using std::max;
using std::min;
using std::floor;
using std::ceil;

namespace mshadow {
template<typename Dtype>
inline void ROIAlignForward(const Tensor<cpu, 4, Dtype> &out,
                           const Tensor<cpu, 4, Dtype> &data,
                           const Tensor<cpu, 2, Dtype> &bbox,
                           const Tensor<cpu, 4, Dtype> &max_idx_x,
                           const Tensor<cpu, 4, Dtype> &max_idx_y,
                           const float spatial_scale_) {
  const Dtype *bottom_data = data.dptr_;
  const Dtype *bottom_rois = bbox.dptr_;
  Dtype *top_data = out.dptr_;
  Dtype *argmax_data_x = max_idx_x.dptr_;
  Dtype *argmax_data_y = max_idx_y.dptr_;
  const int channels_ = data.size(1);
  const int height_ = data.size(2);
  const int width_ = data.size(3);
  const int pooled_height_ = out.size(2);
  const int pooled_width_ = out.size(3);

  const int num_rois = bbox.size(0);
  const int batch_size = data.size(0);
  const int data_size = data.size(1) * data.size(2) * data.size(3);
  // For each ROI R = [batch_index x1 y1 x2 y2]: max pool over R
  for (int n = 0; n < num_rois; ++n) {
    int roi_batch_ind = bottom_rois[0];
    Dtype roi_start_w = bottom_rois[1] * spatial_scale_;
    Dtype roi_start_h = bottom_rois[2] * spatial_scale_;
    Dtype roi_end_w = bottom_rois[3] * spatial_scale_;
    Dtype roi_end_h = bottom_rois[4] * spatial_scale_;
    assert(roi_batch_ind >= 0);
    assert(roi_batch_ind < batch_size);

    // force malformed ROIs to be 1 * 1
    Dtype roi_height = roi_end_h - roi_start_h + 1;
    Dtype roi_width = roi_end_w - roi_start_w + 1;
    const Dtype bin_size_h = static_cast<Dtype>(roi_height)
                             / static_cast<Dtype>(pooled_height_);
    const Dtype bin_size_w = static_cast<Dtype>(roi_width)
                             / static_cast<Dtype>(pooled_width_);

    const Dtype* batch_data = bottom_data + data_size * roi_batch_ind;

    for (int c = 0; c < channels_; ++c) {
      for (int ph = 0; ph < pooled_height_; ++ph) {
        for (int pw = 0; pw < pooled_width_; ++pw) {
          // Compute pooling region for this output unit:
          //  start (included) = floor(ph * roi_height / pooled_height_)
          //  end (excluded) = ceil((ph + 1) * roi_height / pooled_height_)
          Dtype hstart = static_cast<Dtype>(ph) * bin_size_h;
          Dtype wstart = static_cast<Dtype>(pw) * bin_size_w;
          Dtype hend = static_cast<Dtype>(ph + 1) * bin_size_h;
          Dtype wend = static_cast<Dtype>(pw + 1) * bin_size_w;

          hstart = min(max(hstart + roi_start_h, static_cast<Dtype>(0.)), static_cast<Dtype>(height_));
          hend = min(max(hend + roi_start_h, static_cast<Dtype>(0.)), static_cast<Dtype>(height_));
          wstart = min(max(wstart + roi_start_w, static_cast<Dtype>(0.)), static_cast<Dtype>(width_));
          wend = min(max(wend + roi_start_w, static_cast<Dtype>(0.)), static_cast<Dtype>(width_));

          bool is_empty = (hend <= hstart) || (wend <= wstart);

          const int pool_index = ph * pooled_width_ + pw;
          if (is_empty) {
            top_data[pool_index] = 0;
            argmax_data_x[pool_index] = -1;
            argmax_data_y[pool_index] = -1;
          }

          for (Dtype h = hstart; h < hend; h += 1.) {
            for (Dtype w = wstart; w < wend; w += 1.) {
              //const int index = h * width_ + w;
              int x_left = floor(w);
              int x_right = ceil(w);
              int y_bottom = floor(h);
              int y_top = ceil(h);

              const int top_left_index = y_top * width_ + x_left;
              const int top_right_index = y_top * width_ + x_right;
              const int bottom_left_index = y_bottom * width_ + x_left;
              const int bottom_right_index = y_bottom * width_ + x_right;

              //Check whether 4 locations are in bounds
              bool is_top_left_in = x_left >= 0 && x_left <= width_ - 1
                  && y_top >= 0 && y_top <= height_ - 1;
              bool is_top_right_in = x_right >= 0 && x_right <= width_ - 1
                  && y_top >= 0 && y_top <= height_ - 1;
              bool is_bottom_left_in = x_left >= 0 && x_left <= width_ - 1
                  && y_bottom >= 0 && y_bottom <= height_ - 1;
              bool is_bottom_right_in = x_right >= 0 && x_right <= width_ - 1
                  && y_bottom >= 0 && y_bottom <= height_ - 1;

              Dtype val = 0;
              if (is_top_left_in)
                val += (1 - w + x_left) * (h - y_bottom) * batch_data[top_left_index];
              if (is_top_right_in)
                val += (w - x_left) * (h - y_bottom) * batch_data[top_right_index];
              if (is_bottom_left_in)
                val += (1 - w + x_left) * (1 - h + y_bottom) * batch_data[bottom_left_index];
              if (is_bottom_right_in)
                val += (w - x_left) * (1 - h + y_bottom) * batch_data[bottom_right_index];

              if (val > top_data[pool_index]) {
                top_data[pool_index] = val;
                argmax_data_x[pool_index] = w;
                argmax_data_y[pool_index] = h;
              }
            }
          }
        }
      }
      // Increment all data pointers by one channel
      batch_data += data.size(2) * data.size(3);
      top_data += out.size(2) * out.size(3);
      argmax_data_x += max_idx_x.size(2) * max_idx_x.size(3);
      argmax_data_y += max_idx_y.size(2) * max_idx_y.size(3);
    }
    // Increment ROI data pointer
    bottom_rois += bbox.size(1);
  }

  return;
}

template<typename Dtype>
inline void ROIAlignBackwardAcc(const Tensor<cpu, 4, Dtype> &in_grad,
                               const Tensor<cpu, 4, Dtype> &out_grad,
                               const Tensor<cpu, 2, Dtype> &bbox,
                               const Tensor<cpu, 4, Dtype> &max_idx_x,
                               const Tensor<cpu, 4, Dtype> &max_idx_y,
                               const float spatial_scale_) {
  const Dtype *top_diff = out_grad.dptr_;
  const Dtype *bottom_rois = bbox.dptr_;
  Dtype *bottom_diff = in_grad.dptr_;
  Dtype *argmax_data_x = max_idx_x.dptr_;
  Dtype *argmax_data_y = max_idx_y.dptr_;

  const int batch_size_ = in_grad.size(0);
  const int channels_ = in_grad.size(1);
  const int height_ = in_grad.size(2);
  const int width_ = in_grad.size(3);
  const int pooled_height_ = out_grad.size(2);
  const int pooled_width_ = out_grad.size(3);

  const int num_rois = bbox.size(0);

  for (int b = 0; b < batch_size_; ++b) {
    for (int c = 0; c < channels_; ++c) {
      for (int h = 0; h < height_; ++h) {
        for (int w = 0; w < width_; ++w) {
          int offset_bottom_diff = (b * channels_ + c) * height_ * width_;
          offset_bottom_diff += h * width_ + w;

          Dtype gradient = 0;
          // Accumulate gradient over all ROIs that pooled this element
          for (int roi_n = 0; roi_n < num_rois; ++roi_n) {
            const Dtype* offset_bottom_rois = bottom_rois + roi_n * 5;
            int roi_batch_ind = offset_bottom_rois[0];
            assert(roi_batch_ind >= 0);
            assert(roi_batch_ind < batch_size_);
            if (b != roi_batch_ind) {
              continue;
            }

            int roi_start_w = floor(offset_bottom_rois[1] * spatial_scale_);
            int roi_start_h = floor(offset_bottom_rois[2] * spatial_scale_);
            int roi_end_w = ceil(offset_bottom_rois[3] * spatial_scale_);
            int roi_end_h = ceil(offset_bottom_rois[4] * spatial_scale_);

            bool in_roi = (w >= roi_start_w && w <= roi_end_w &&
                           h >= roi_start_h && h <= roi_end_h);
            if (!in_roi) {
              continue;
            }

            // force malformed ROIs to be 1 * 1
            Dtype roi_height = roi_end_h - roi_start_h + 1;
            Dtype roi_width = roi_end_w - roi_start_w + 1;
            const Dtype bin_size_h = static_cast<Dtype>(roi_height)
                                     / static_cast<Dtype>(pooled_height_);
            const Dtype bin_size_w = static_cast<Dtype>(roi_width)
                                     / static_cast<Dtype>(pooled_width_);

            // compute pooled regions correspond to original (h, w) point
            int phstart = floor(static_cast<Dtype>(h - roi_start_h) / bin_size_h);
            int pwstart = floor(static_cast<Dtype>(w - roi_start_w) / bin_size_w);
            int phend = ceil(static_cast<Dtype>(h - roi_start_h + 1) / bin_size_h);
            int pwend = ceil(static_cast<Dtype>(w - roi_start_w + 1) / bin_size_w);

            // clip to boundaries of pooled region
            phstart = min(max(phstart, 0), pooled_height_);
            phend = min(max(phend, 0), pooled_height_);
            pwstart = min(max(pwstart, 0), pooled_width_);
            pwend = min(max(pwend, 0), pooled_width_);

            // accumulate over gradients in pooled regions
            int offset = (roi_n * channels_ + c) * pooled_height_ * pooled_width_;
            const Dtype* offset_top_diff = top_diff + offset;
            const Dtype* offset_argmax_data_x = argmax_data_x + offset;
            const Dtype* offset_argmax_data_y = argmax_data_y + offset;

            for (int ph = phstart; ph < phend; ++ph) {
              for (int pw = pwstart; pw < pwend; ++pw) {
                const int pooled_index = ph * pooled_width_ + pw;
                Dtype max_x = offset_argmax_data_x[pooled_index];
                Dtype max_y = offset_argmax_data_y[pooled_index];

                int x_left = floor(max_x);
                int x_right = ceil(max_x);
                int y_bottom = floor(max_y);
                int y_top = ceil(max_y);

                // if (w,h) is 1 location of the 4 bilinear locations， it can get gradient
                if (x_left == w && y_top == h)
                  gradient += (1 - max_x + x_left) * (1 + max_y - y_top)
                      * offset_top_diff[pooled_index];
                else if (x_left == w && y_bottom == h)
                  gradient += (1 - max_x + x_left) * (1 - max_y + y_bottom)
                      * offset_top_diff[pooled_index];
                else if (x_right == w && y_top == h)
                  gradient += (max_x - x_left) * (1 + max_y - y_top)
                      * offset_top_diff[pooled_index];
                else if (x_right == w && y_bottom == h)
                  gradient += (max_x - x_left) * (1 - max_y + y_bottom)
                      * offset_top_diff[pooled_index];

              }
            }
          }
          bottom_diff[offset_bottom_diff] += gradient;
        }
      }
    }
  }

  return;
}
}  // namespace mshadow

namespace mxnet {
namespace op {

template<>
Operator *CreateOp<cpu>(ROIAlignParam param, int dtype) {
  Operator* op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new ROIAlignOp<cpu, DType>(param);
  });
  return op;
}

Operator *ROIAlignProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                           std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, param_, in_type->at(0));
}

DMLC_REGISTER_PARAMETER(ROIAlignParam);

MXNET_REGISTER_OP_PROPERTY(ROIAlign, ROIAlignProp)
.describe("Performs region of interest(ROI) align on the input array.")
.add_argument("data", "NDArray-or-Symbol", "The input array to the pooling operator, "
                                            " a 4D Feature maps ")
.add_argument("rois", "NDArray-or-Symbol", "Bounding box coordinates, a 2D array of "
"[[batch_index, x1, y1, x2, y2]], where (x1, y1) and (x2, y2) are top left and bottom right "
"corners of designated region of interest. `batch_index` indicates the index of corresponding "
"image in the input array")
.add_arguments(ROIAlignParam::__FIELDS__());
}  // namespace op
}  // namespace mxnet
