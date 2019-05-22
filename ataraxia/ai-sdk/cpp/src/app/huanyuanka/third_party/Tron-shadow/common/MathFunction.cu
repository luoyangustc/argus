#include "MathFunction.hpp"

namespace Shadow
{

__global__ void processNN_kernel(int memnum, const cv::cuda::PtrStepSz<uchar3> src, float *dst,
                                    const float fx, const float fy,
                                    const int newWidth, const int newHeight,
                                    const float b_mean, const float g_mean, const float r_mean, const float scale)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < newWidth && y < newHeight)
    {
        const int src_x = (int)(x * fx + 0.5);
        const int src_y = (int)(y * fy + 0.5);
        uchar3 v = src(src_y, src_x);
        dst[y * newHeight + x] = ((float)v.x - b_mean) * scale;
        dst[memnum + y * newHeight + x] = ((float)v.y - g_mean) * scale;
        dst[2 * memnum + y * newHeight + x] = ((float)v.z - r_mean) * scale;
    }
}

__global__ void processBI_kernel(int memnum, const cv::cuda::PtrStepSz<uchar3> src, float *dst,
                                    const float scale_x, const float scale_y,
                                    const int newWidth, const int newHeight,
                                    const float b_mean, const float g_mean, const float r_mean, const float scale)
{

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    float fy = (float)((y + 0.5) * scale_y - 0.5);
    int sy = floor(fy);
    fy -= sy;
    sy = sy < src.rows - 2 ? sy : src.rows - 2;
    sy = 0 > sy ? 0 : sy;

    float fx = (float)((x + 0.5) * scale_x - 0.5);
    int sx = floor(fx);
    fx -= sx;
    sx = sx < src.cols - 2 ? sx : src.cols - 2;
    sx = 0 > sx ? 0 : sx;

    dst[y * newHeight + x] = ((1 - fx) * (1 - fy) * src(sy, sx).x +
                              (1 - fx) * fy * src(sy + 1, sx).x +
                              fx * (1 - fy) * src(sy, sx + 1).x +
                              fx * fy * src(sy + 1, sx + 1).x - b_mean) *
                             scale;
    dst[memnum + y * newHeight + x] = ((1 - fx) * (1 - fy) * src(sy, sx).y +
                                       (1 - fx) * fy * src(sy + 1, sx).y +
                                       fx * (1 - fy) * src(sy, sx + 1).y +
                                       fx * fy * src(sy + 1, sx + 1).y - g_mean) *
                                      scale;
    dst[2 * memnum + y * newHeight + x] = ((1 - fx) * (1 - fy) * src(sy, sx).z +
                                           (1 - fx) * fy * src(sy + 1, sx).z +
                                           fx * (1 - fy) * src(sy, sx + 1).z +
                                           fx * fy * src(sy + 1, sx + 1).z - r_mean) *
                                          scale;
}

void process_caller(const cv::cuda::PtrStepSz<uchar3> &src, float *dst,
                       const int newWidth, const int newHeight,
                       float *param, InterMethod interMethod)
{
    int memnum = newWidth * newHeight;
    const float fx = (float)src.cols / newWidth;
    const float fy = (float)src.rows / newHeight;
    const float b_mean = param[0], g_mean = param[1], r_mean = param[2], scale = param[3];
    dim3 block(GET_BLOCKS(newWidth), GET_BLOCKS(newHeight));
    dim3 grid((newWidth + block.x - 1) / block.x, (newHeight + block.y - 1) / block.y);
    if (interMethod == nearest)
        processNN_kernel<<<grid, block>>>(memnum, src, dst, fx, fy, newWidth, newHeight, b_mean, g_mean, r_mean, scale);
    if (interMethod == bilinear)
        processBI_kernel<<<grid, block>>>(memnum, src, dst, fx, fy, newWidth, newHeight, b_mean, g_mean, r_mean, scale);
    //cudaDeviceSynchronize();
}

void process(const cv::cuda::GpuMat &src, float *dst,
                const int newWidth, const int newHeight,
                float *param, InterMethod interMethod)
{
    //cv::cuda::Stream& stream = cv::cuda::Stream::Null();
    //cudaStream_t s = cv::cuda::StreamAccessor::getStream(stream);
    cv::cuda::PtrStepSz<uchar3> ptr = src;
    process_caller(ptr, dst, newWidth, newHeight, param, interMethod);
}

} // namespace Shadow
