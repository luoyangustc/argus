#ifndef __MATHFUNCTION_HPP__
#define __MATHFUNCTION_HPP__

#include "cuda_runtime.h"
#include "Net.hpp"
#include <opencv2/opencv.hpp>

namespace Shadow
{
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)

// CUDA: use 16*16<512 threads per block
const int CUDA_NUM_THREADS = 512;

// CUDA: number of blocks for threads.
inline int GET_BLOCKS(const int N)
{
    return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

void process(const cv::cuda::GpuMat &src, float *dst,
             const int new_w, const int new_h,
             float *param, InterMethod interMethod);

} // namespace Shadow
#endif
