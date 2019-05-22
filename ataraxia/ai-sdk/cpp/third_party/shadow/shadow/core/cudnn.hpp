#ifndef SHADOW_CORE_CUDNN_HPP
#define SHADOW_CORE_CUDNN_HPP

#include "util/log.hpp"

#if defined(USE_CUDNN)
#include "cudnn.h"
#endif

namespace Shadow {

#if defined(USE_CUDNN)

#define CUDNN_VERSION_MIN(major, minor, patch) \
  (CUDNN_VERSION >= (major * 1000 + minor * 100 + patch))

#define CUDNN_CHECK(condition)                 \
  do {                                         \
    cudnnStatus_t status = condition;          \
    CHECK_EQ(status, CUDNN_STATUS_SUCCESS)     \
        << " " << cudnnGetErrorString(status); \
  } while (0)

namespace cudnn {

template <typename Dtype>
class dataType;
template <>
class dataType<float> {
 public:
  static const cudnnDataType_t type = CUDNN_DATA_FLOAT;
  static float oneval, zeroval;
  static const void *one, *zero;
};
template <>
class dataType<double> {
 public:
  static const cudnnDataType_t type = CUDNN_DATA_DOUBLE;
  static double oneval, zeroval;
  static const void *one, *zero;
};

template <typename Dtype>
inline void createTensorDesc(cudnnTensorDescriptor_t* desc) {
  CUDNN_CHECK(cudnnCreateTensorDescriptor(desc));
}

template <typename Dtype>
inline void setTensor4dDesc(cudnnTensorDescriptor_t* desc, int n, int c, int h,
                            int w) {
  CUDNN_CHECK(cudnnSetTensor4dDescriptor(*desc, CUDNN_TENSOR_NCHW,
                                         dataType<Dtype>::type, n, c, h, w));
}

template <typename Dtype>
inline void createFilterDesc(cudnnFilterDescriptor_t* desc) {
  CUDNN_CHECK(cudnnCreateFilterDescriptor(desc));
}

template <typename Dtype>
inline void setFilter4dDesc(cudnnFilterDescriptor_t* desc, int n, int c, int h,
                            int w) {
  CUDNN_CHECK(cudnnSetFilter4dDescriptor(*desc, dataType<Dtype>::type,
                                         CUDNN_TENSOR_NCHW, n, c, h, w));
}

template <typename Dtype>
inline void createConvolutionDesc(cudnnConvolutionDescriptor_t* conv_desc) {
  CUDNN_CHECK(cudnnCreateConvolutionDescriptor(conv_desc));
}

template <typename Dtype>
inline void setConvolution2dDesc(cudnnConvolutionDescriptor_t* conv_desc,
                                 int pad_h, int pad_w, int stride_h,
                                 int stride_w, int dilation_h, int dilation_w,
                                 int group) {
  CUDNN_CHECK(cudnnSetConvolution2dDescriptor(
      *conv_desc, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
      CUDNN_CROSS_CORRELATION, dataType<Dtype>::type));
#if CUDNN_VERSION_MIN(7, 0, 1)
  CUDNN_CHECK(cudnnSetConvolutionGroupCount(*conv_desc, group));
#endif
}

template <typename Dtype>
inline void createActivationDesc(cudnnActivationDescriptor_t* activate_desc) {
  CUDNN_CHECK(cudnnCreateActivationDescriptor(activate_desc));
}

template <typename Dtype>
inline void setActivationDesc(cudnnActivationDescriptor_t* activate_desc,
                              int activate_type, double coef) {
  cudnnActivationMode_t mode{};
  switch (activate_type) {
#if CUDNN_VERSION_MIN(7, 1, 1)
    case -1:
      mode = CUDNN_ACTIVATION_IDENTITY;
      break;
#endif
    case 1:
      mode = CUDNN_ACTIVATION_RELU;
      break;
    case 2:
      mode = CUDNN_ACTIVATION_CLIPPED_RELU;
      break;
    case 3:
      mode = CUDNN_ACTIVATION_SIGMOID;
      break;
    case 5:
      mode = CUDNN_ACTIVATION_TANH;
      break;
    default:
      LOG(FATAL) << "Unsupported activate type " << activate_type;
  }
  CUDNN_CHECK(cudnnSetActivationDescriptor(*activate_desc, mode,
                                           CUDNN_NOT_PROPAGATE_NAN, coef));
}

template <typename Dtype>
inline void createPoolingDesc(cudnnPoolingDescriptor_t* pool_desc) {
  CUDNN_CHECK(cudnnCreatePoolingDescriptor(pool_desc));
}

template <typename Dtype>
inline void setPooling2dDesc(cudnnPoolingDescriptor_t* pool_desc, int pool_type,
                             int window_h, int window_w, int pad_h, int pad_w,
                             int stride_h, int stride_w) {
  cudnnPoolingMode_t mode{};
  switch (pool_type) {
    case 0:
      mode = CUDNN_POOLING_MAX;
      break;
    case 1:
      mode = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
      break;
    default:
      LOG(FATAL) << "Unsupported pool type " << pool_type;
  }
  CUDNN_CHECK(cudnnSetPooling2dDescriptor(*pool_desc, mode, CUDNN_PROPAGATE_NAN,
                                          window_h, window_w, pad_h, pad_w,
                                          stride_h, stride_w));
}

}  // namespace cudnn

namespace Kernel {

extern cudnnHandle_t cudnn_handle_;

}  // namespace Kernel

#endif

}  // namespace Shadow

#endif  // SHADOW_CORE_CUDNN_HPP
