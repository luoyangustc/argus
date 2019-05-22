/*!
 * Copyright (c) 2015 by Contributors
 * \file roi_pooling.cu
 * \brief roi pooling operator
 * \author Ross Girshick, Kye-Hyeon Kim, Jian Guo
 * \changed to roi_align by Elaine Bao
 * \file roi_align.cu
 * \roi align operator described in Mask RCNN
*/
#include "./roi_align-inl.h"
#include <mshadow/tensor.h>
#include <mshadow/cuda/reduce.cuh>
#include <algorithm>
#include <vector>

const int kMaxGridDim = 65535;

namespace mshadow {
namespace cuda {


template<typename Dtype>
__global__ void ROIAlignForwardKernel(const int count, const Dtype* bottom_data,
                                     const float spatial_scale, const int channels,
                                     const int height, const int width,
                                     const int pooled_height, const int pooled_width,
                                     const Dtype* bottom_rois, Dtype* top_data,
                                     Dtype* argmax_data_x, Dtype* argmax_data_y) {
  for (int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
       index < count;
       index += blockDim.x * gridDim.x * gridDim.y) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    bottom_rois += n * 5;
    int roi_batch_ind = bottom_rois[0];

    if (roi_batch_ind < 0) {
      top_data[index] = 0;
      argmax_data_x[index] = 0;
      argmax_data_y[index] = 0;
      continue;
    }

    Dtype roi_start_w = bottom_rois[1] * spatial_scale;
    Dtype roi_start_h = bottom_rois[2] * spatial_scale;
    Dtype roi_end_w = bottom_rois[3] * spatial_scale;
    Dtype roi_end_h = bottom_rois[4] * spatial_scale;

    // Force malformed ROIs to be 1x1
    Dtype roi_width = roi_end_w - roi_start_w + 1;
    Dtype roi_height = roi_end_h - roi_start_h + 1;
    Dtype bin_size_h = static_cast<Dtype>(roi_height)
                       / static_cast<Dtype>(pooled_height);
    Dtype bin_size_w = static_cast<Dtype>(roi_width)
                       / static_cast<Dtype>(pooled_width);

    Dtype hstart = static_cast<Dtype>(ph) * bin_size_h;
    Dtype wstart = static_cast<Dtype>(pw) * bin_size_w;
    Dtype hend = static_cast<Dtype>(ph + 1) * bin_size_h;
    Dtype wend = static_cast<Dtype>(pw + 1) * bin_size_w;

    // Add roi offsets and clip to input boundaries
    hstart = min(max(hstart + roi_start_h, 0.), static_cast<Dtype>(height));
    hend = min(max(hend + roi_start_h, 0.), static_cast<Dtype>(height));
    wstart = min(max(wstart + roi_start_w, 0.), static_cast<Dtype>(width));
    wend = min(max(wend + roi_start_w, 0.), static_cast<Dtype>(width));
    bool is_empty = (hend <= hstart) || (wend <= wstart);

    // Define an empty pooling region to be zero
    Dtype maxval = is_empty ? 0 : -FLT_MAX;
    // If nothing is pooled, argmax = -1 causes nothing to be backprop'd
    Dtype maxidx_x = -1;
    Dtype maxidx_y = -1;
    bottom_data += (roi_batch_ind * channels + c) * height * width;
    for (Dtype h = hstart; h < hend; h += 1.) {
      for (Dtype w = wstart; w < wend; w += 1.) {
        // Selecting four regular locations for bilinear interpolation
        int x_left = floor(w);
        int x_right = ceil(w);
        int y_bottom = floor(h);
        int y_top = ceil(h);

        int top_left_index = y_top * width + x_left;
        int top_right_index = y_top * width + x_right;
        int bottom_left_index = y_bottom * width + x_left;
        int bottom_right_index = y_bottom * width + x_right;

        //Check whether 4 locations are in bounds
        bool is_top_left_in = x_left >= 0 && x_left <= width - 1
            && y_top >= 0 && y_top <= height - 1;
        bool is_top_right_in = x_right >= 0 && x_right <= width - 1
            && y_top >= 0 && y_top <= height - 1;
        bool is_bottom_left_in = x_left >= 0 && x_left <= width - 1
            && y_bottom >= 0 && y_bottom <= height - 1;
        bool is_bottom_right_in = x_right >= 0 && x_right <= width - 1
            && y_bottom >= 0 && y_bottom <= height - 1;

        //do bilinear interpolation
        Dtype val = 0;
        if (is_top_left_in)
          val += (1 - w + x_left) * (h - y_bottom) * bottom_data[top_left_index];
        if (is_top_right_in)
          val += (w - x_left) * (h - y_bottom) * bottom_data[top_right_index];
        if (is_bottom_left_in)
          val += (1 - w + x_left) * (1 - h + y_bottom) * bottom_data[bottom_left_index];
        if (is_bottom_right_in)
          val += (w - x_left) * (1 - h + y_bottom) * bottom_data[bottom_right_index];

        if (val > maxval){
          maxval = val;
          maxidx_x = w;
          maxidx_y = h;
        }
      }
    }
    top_data[index] = maxval;
    argmax_data_x[index] = maxidx_x;
    argmax_data_y[index] = maxidx_y;
  }
}

template<typename Dtype>
inline void ROIAlignForward(const Tensor<gpu, 4, Dtype> &out,
                           const Tensor<gpu, 4, Dtype> &data,
                           const Tensor<gpu, 2, Dtype> &bbox,
                           const Tensor<gpu, 4, Dtype> &max_idx_x,
                           const Tensor<gpu, 4, Dtype> &max_idx_y,
                           const float spatial_scale) {
  const Dtype *bottom_data = data.dptr_;
  const Dtype *bottom_rois = bbox.dptr_;
  Dtype *top_data = out.dptr_;
  Dtype *argmax_data_x = max_idx_x.dptr_;
  Dtype *argmax_data_y = max_idx_y.dptr_;
  const int count = out.shape_.Size();
  const int channels = data.size(1);
  const int height = data.size(2);
  const int width = data.size(3);
  const int pooled_height = out.size(2);
  const int pooled_width = out.size(3);
  const int gridSize = (count + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock;
  dim3 dimGrid(kMaxGridDim, (gridSize + kMaxGridDim - 1) / kMaxGridDim);
  dim3 dimBlock(kMaxThreadsPerBlock);
  CheckLaunchParam(dimGrid, dimBlock, "ROIAlign Forward");
  cudaStream_t stream = Stream<gpu>::GetStream(out.stream_);
  ROIAlignForwardKernel<Dtype><<<dimGrid, dimBlock, 0, stream>>>(
      count, bottom_data, spatial_scale, channels, height, width,
      pooled_height, pooled_width, bottom_rois, top_data, argmax_data_x, argmax_data_y);
}

template<typename Dtype>
__global__ void ROIAlignBackwardAccKernel(const int count, const Dtype* top_diff,
                                         const Dtype* argmax_data_x,
                                         const Dtype* argmax_data_y,
                                         const int num_rois,
                                         const float spatial_scale, const int channels,
                                         const int height, const int width,
                                         const int pooled_height, const int pooled_width,
                                         Dtype* bottom_diff, const Dtype* bottom_rois) {
  for (int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
       index < count;
       index += blockDim.x * gridDim.x * gridDim.y) {
    // (n, c, h, w) coords in bottom data
    int w = index % width;
    int h = (index / width) % height;
    int c = (index / width / height) % channels;
    int n = index / width / height / channels;

    Dtype gradient = 0;
    // Accumulate gradient over all ROIs that pooled this element
    for (int roi_n = 0; roi_n < num_rois; ++roi_n) {
      const Dtype* offset_bottom_rois = bottom_rois + roi_n * 5;
      int roi_batch_ind = offset_bottom_rois[0];
      // Skip if ROI's batch index doesn't match n
      if (n != roi_batch_ind) {
        continue;
      }

      // And it assumes that we don't have any negative offset of course
      int roi_start_w = floor(offset_bottom_rois[1] * spatial_scale);
      int roi_start_h = floor(offset_bottom_rois[2] * spatial_scale);
      int roi_end_w = ceil(offset_bottom_rois[3] * spatial_scale);
      int roi_end_h = ceil(offset_bottom_rois[4] * spatial_scale);

      // Skip if ROI doesn't include (h, w)
      const bool in_roi = (w >= roi_start_w && w <= roi_end_w &&
                           h >= roi_start_h && h <= roi_end_h);
      if (!in_roi) {
        continue;
      }

      int offset = (roi_n * channels + c) * pooled_height * pooled_width;
      const Dtype* offset_top_diff = top_diff + offset;
      const Dtype* offset_argmax_data_x = argmax_data_x + offset;
      const Dtype* offset_argmax_data_y = argmax_data_y + offset;

      // Compute feasible set of pooled units that could have pooled
      // this bottom unit
      Dtype roi_width = roi_end_w - roi_start_w + 1;
      Dtype roi_height = roi_end_h - roi_start_h + 1;

      Dtype bin_size_h = static_cast<Dtype>(roi_height)
                         / static_cast<Dtype>(pooled_height);
      Dtype bin_size_w = static_cast<Dtype>(roi_width)
                         / static_cast<Dtype>(pooled_width);

      int phstart = floor(static_cast<Dtype>(h - roi_start_h) / bin_size_h);
      int phend = ceil(static_cast<Dtype>(h - roi_start_h + 1) / bin_size_h);
      int pwstart = floor(static_cast<Dtype>(w - roi_start_w) / bin_size_w);
      int pwend = ceil(static_cast<Dtype>(w - roi_start_w + 1) / bin_size_w);

      phstart = min(max(phstart, 0), pooled_height);
      phend = min(max(phend, 0), pooled_height);
      pwstart = min(max(pwstart, 0), pooled_width);
      pwend = min(max(pwend, 0), pooled_width);

      for (int ph = phstart; ph < phend; ++ph) {
        for (int pw = pwstart; pw < pwend; ++pw) {
          int index = ph * pooled_width + pw;
          Dtype max_x = offset_argmax_data_x[index];
          Dtype max_y = offset_argmax_data_y[index];

          int x_left = floor(max_x);
          int x_right = ceil(max_x);
          int y_bottom = floor(max_y);
          int y_top = ceil(max_y);

          // if (w,h) is 1 location of the 4 bilinear locations， it can get gradient
          if (x_left == w && y_top == h)
            gradient += (1 - max_x + x_left) * (1 + max_y - y_top)
                * offset_top_diff[index];
          else if (x_left == w && y_bottom == h)
            gradient += (1 - max_x + x_left) * (1 - max_y + y_bottom)
                * offset_top_diff[index];
          else if (x_right == w && y_top == h)
            gradient += (max_x - x_left) * (1 + max_y - y_top)
                * offset_top_diff[index];
          else if (x_right == w && y_bottom == h)
            gradient += (max_x - x_left) * (1 - max_y + y_bottom)
                * offset_top_diff[index];
        }
      }
    }
    bottom_diff[index] += gradient;
  }
}

template<typename Dtype>
inline void ROIAlignBackwardAcc(const Tensor<gpu, 4, Dtype> &in_grad,
                               const Tensor<gpu, 4, Dtype> &out_grad,
                               const Tensor<gpu, 2, Dtype> &bbox,
                               const Tensor<gpu, 4, Dtype> &max_idx_x,
                               const Tensor<gpu, 4, Dtype> &max_idx_y,
                               const float spatial_scale) {
  const Dtype *top_diff = out_grad.dptr_;
  const Dtype *bottom_rois = bbox.dptr_;
  Dtype *bottom_diff = in_grad.dptr_;
  Dtype *argmax_data_x = max_idx_x.dptr_;
  Dtype *argmax_data_y = max_idx_y.dptr_;
  const int count = in_grad.shape_.Size();
  const int num_rois = bbox.size(0);
  const int channels = in_grad.size(1);
  const int height = in_grad.size(2);
  const int width = in_grad.size(3);
  const int pooled_height = out_grad.size(2);
  const int pooled_width = out_grad.size(3);
  const int gridSize = (count + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock;
  dim3 dimGrid(kMaxGridDim, (gridSize + kMaxGridDim - 1) / kMaxGridDim);
  dim3 dimBlock(kMaxThreadsPerBlock);
  CheckLaunchParam(dimGrid, dimBlock, "ROIAlign Backward");
  cudaStream_t stream = Stream<gpu>::GetStream(in_grad.stream_);
  ROIAlignBackwardAccKernel<Dtype><<<dimGrid, dimBlock, 0, stream>>>(
      count, top_diff, argmax_data_x, argmax_data_y, num_rois, spatial_scale, channels, height, width,
      pooled_height, pooled_width, bottom_diff, bottom_rois);
}

}  // namespace cuda

template<typename Dtype>
inline void ROIAlignForward(const Tensor<gpu, 4, Dtype> &out,
                           const Tensor<gpu, 4, Dtype> &data,
                           const Tensor<gpu, 2, Dtype> &bbox,
                           const Tensor<gpu, 4, Dtype> &max_idx_x,
                           const Tensor<gpu, 4, Dtype> &max_idx_y,
                           const float spatial_scale) {
  cuda::ROIAlignForward(out, data, bbox, max_idx_x, max_idx_y, spatial_scale);
}

template<typename Dtype>
inline void ROIAlignBackwardAcc(const Tensor<gpu, 4, Dtype> &in_grad,
                               const Tensor<gpu, 4, Dtype> &out_grad,
                               const Tensor<gpu, 2, Dtype> &bbox,
                               const Tensor<gpu, 4, Dtype> &max_idx_x,
                               const Tensor<gpu, 4, Dtype> &max_idx_y,
                               const float spatial_scale) {
  cuda::ROIAlignBackwardAcc(in_grad, out_grad, bbox, max_idx_x, max_idx_y, spatial_scale);
}

}  // namespace mshadow


namespace mxnet {
namespace op {

/*
NNVM_REGISTER_OP(ROIAlign)
.set_attr<FCompute>("FCompute<gpu>", ROIAlignForward<gpu>);

NNVM_REGISTER_OP(_backward_ROIAlign)
.set_attr<FCompute>("FCompute<gpu>", ROIAlignBackwardAcc<gpu>);
*/

template<>
Operator* CreateOp<gpu>(ROIAlignParam param, int dtype) {
  Operator* op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new ROIAlignOp<gpu, DType>(param);
  });
  return op;
}

}  // namespace op
}  // namespace mxnet
