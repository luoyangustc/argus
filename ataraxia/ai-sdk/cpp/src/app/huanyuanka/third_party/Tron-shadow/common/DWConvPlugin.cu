#include "DWConvPlugin.hpp"
#include "MathFunction.hpp"

namespace Shadow
{

__global__ void dw_conv_kernel(const int nthreads, const float* bottom_data, const int channels,
							   const int height, const int width, const int conved_height, const int conved_width,
							   const int kernel_h, const int kernel_w, const int stride_h, const int stride_w, const int pad_h, const int pad_w,
							   float *top_data, const float *weight, const float *bias, const bool bias_term_)
{
	CUDA_KERNEL_LOOP(index, nthreads)
	{
		const int pw = index % conved_width;
		const int ph = (index / conved_width) % conved_height;
		const int c = (index / conved_width / conved_height) % channels;
		const int n = index / conved_width / conved_height / channels;
		int hstart = ph * stride_h - pad_h;
		int wstart = pw * stride_w - pad_w;
		int hend = min(hstart + kernel_h, height + pad_h);
		int wend = min(wstart + kernel_w, width + pad_w);
		hstart = max(hstart, 0);
		wstart = max(wstart, 0);
		hend = min(hend, height);
		wend = min(wend, width);
		float aveval = 0;
		const float *bottom_slice =
			bottom_data + (n * channels + c) * height * width;
		const float *weight_slice =
			weight + c * kernel_h * kernel_w;
		int khstart = hend < kernel_h ? kernel_h - hend : 0;
		int kwstart = wend < kernel_w ? kernel_w - wend : 0;
		for (int h = hstart; h < hend; ++h)
		{
			for (int w = wstart; w < wend; ++w)
			{
				aveval += bottom_slice[h * width + w] * weight_slice[(khstart + h - hstart) * kernel_w + (kwstart + w - wstart)];
			}
		}
		if (bias_term_)
		{
			aveval += bias[c];
		}
		top_data[index] = aveval;
	}
}

void DWConvLayer(int batch_size, const float *input_data, float *output_data, const float *kernel, KernelInfo kernelInfo,
				 DimsCHW inputShape, DimsCHW outputShape, bool bias_term, const float *bias)
{
	const int channels = inputShape.c();
	const int height = inputShape.h();
	const int width = inputShape.w();
	const int conved_height = outputShape.h();
	const int conved_width = outputShape.w();

	const int kernel_h = kernelInfo.kernelSize[0];
	const int kernel_w = kernelInfo.kernelSize[1];
	const int stride_h = kernelInfo.stride[0];
	const int stride_w = kernelInfo.stride[1];
	const int pad_h = kernelInfo.pad[0];
	const int pad_w = kernelInfo.pad[1];
	const int count = batch_size * outputShape.c() * outputShape.h() * outputShape.w();

	dw_conv_kernel<<<GET_BLOCKS(count), CUDA_NUM_THREADS>>>(count, input_data, channels, height, width, conved_height, conved_width,
															kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, output_data, kernel, bias, bias_term);
	cudaDeviceSynchronize();
}

} // namespace Shadow
