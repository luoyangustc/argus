#include "SlicePlugin.hpp"
#include "MathFunction.hpp"

namespace Shadow
{

__global__ void slice_kernel(int nthreads, const float* input_data, float* output_data, 
                            int slice_size, int input_slice_axis, 
                            int output_slice_axis, int offset_slice_axis)
{
    CUDA_KERNEL_LOOP(index, nthreads) {
        const int total_slice_size = slice_size * output_slice_axis;
        const int slice_num = index / total_slice_size;
        const int slice_index = index % total_slice_size;
        const int bottom_index = slice_index + (slice_num * input_slice_axis + offset_slice_axis) * slice_size;
        output_data[index] = input_data[bottom_index];
    }
}

void SliceLayer(int nthreads, const float* input_data, float* output_data, 
                int slice_size, int input_slice_axis, 
                int output_slice_axis, int offset_slice_axis)
{
    slice_kernel<<<GET_BLOCKS(nthreads), CUDA_NUM_THREADS>>>
    (nthreads, input_data, output_data, slice_size, input_slice_axis, output_slice_axis, offset_slice_axis);
    //cudaDeviceSynchronize();
}

} // namespace Shadow
