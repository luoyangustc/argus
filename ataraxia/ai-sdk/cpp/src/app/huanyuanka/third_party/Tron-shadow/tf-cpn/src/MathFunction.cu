#include "MathFunction.hpp"

    __global__ void processBI_kernel(float *src, float *dst, int oldWidth, int oldHeight, int newWidth, int newHeight)
    {
        const float scale_x = (float)oldWidth / newWidth;
        const float scale_y = (float)oldHeight / newHeight;
        
        int x = threadIdx.x + blockIdx.x * blockDim.x;
        int y = threadIdx.y + blockIdx.y * blockDim.y;
        
        float fy = (float)((y + 0.5) * scale_y - 0.5);
        int sy = floor(fy);
        fy -= sy;
        sy = sy < oldHeight - 2 ? sy : oldHeight - 2;
        sy = 0 > sy ? 0 : sy;
        
        float fx = (float)((x + 0.5) * scale_x - 0.5);
        int sx = floor(fx);
        fx -= sx;
        sx = sx < oldWidth - 2 ? sx : oldWidth - 2;
        sx = 0 > sx ? 0 : sx;
        float pixel1,pixel2,pixel3,pixel4;
        pixel1 = src[(sy)*3+(sx)*3];
        pixel2 = src[(sy+1)*3+(sx)*3];
        pixel3 = src[(sy)*3+(sx+1)*3];
        pixel4 = src[(sy+1)*3+(sx+1)*3];
        dst[y * newHeight + x] = ((1 - fx) * (1 - fy) * pixel1 +
                                  (1 - fx) * fy * pixel2 +
                                  fx * (1 - fy) * pixel3 +
                                  fx * fy * pixel4);
        pixel1 = src[(sy)*3+(sx)*3+1];
        pixel2 = src[(sy+1)*3+(sx)*3+1];
        pixel3 = src[(sy)*3+(sx+1)*3+1];
        pixel4 = src[(sy+1)*3+(sx+1)*3+1];
        dst[1 + y * newHeight + x] = ((1 - fx) * (1 - fy) * pixel1 +
                                           (1 - fx) * fy * pixel2 +
                                           fx * (1 - fy) * pixel3 +
                                           fx * fy * pixel4);
        pixel1 = src[(sy)*3+(sx)*3+2];
        pixel2 = src[(sy+1)*3+(sx)*3+2];
        pixel3 = src[(sy)*3+(sx+1)*3+2];
        pixel4 = src[(sy+1)*3+(sx+1)*3+2];
        dst[2 + y * newHeight + x] = ((1 - fx) * (1 - fy) * pixel1 +
                                               (1 - fx) * fy * pixel2 +
                                               fx * (1 - fy) * pixel3 +
                                               fx * fy * pixel4);
    }
    
void process_caller(float *src, float *dst,
                    int oldWidth, int oldHeight,
                        int newWidth, int newHeight)
    {
        dim3 block(GET_BLOCKS(newWidth), GET_BLOCKS(newHeight));
        dim3 grid((newWidth + block.x - 1) / block.x, (newHeight + block.y - 1) / block.y);
        
        processBI_kernel<<<grid, block>>>(src, dst, oldWidth, oldHeight, newWidth, newHeight);
        //cudaDeviceSynchronize();
    }
    
void process(float *src, float *dst,
             int oldWidth, int oldHeight,
                 int newWidth, int newHeight)
    {

        process_caller(src, dst, oldWidth, oldHeight, newWidth, newHeight);
    }
    
