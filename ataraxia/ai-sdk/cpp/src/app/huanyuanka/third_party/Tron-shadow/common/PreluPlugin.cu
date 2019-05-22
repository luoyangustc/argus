#include "PreluPlugin.hpp"
#include "MathFunction.hpp"
namespace Shadow
{

__global__ void PReLU(const int n, const int channels, const int dim,
    const float* in, float* out, const float* slope_data, const int div_factor) {
  CUDA_KERNEL_LOOP(index, n) {
    int c = (index / dim) % channels / div_factor;
    out[index] = in[index] > 0 ? in[index] : in[index] * slope_data[c];
  }
}


void PreluLayer(const int count, const int channels, const int dim, const float* bottom_data,
  float* top_data, void* mDeviceKernel, const int div_factor){
  PReLU<<<GET_BLOCKS(count), CUDA_NUM_THREADS>>>(count,channels,dim,bottom_data,top_data,static_cast<const float*>(mDeviceKernel),div_factor);
}

}//namespace Shadow
