#ifndef __MATHFUNCTION_H__
#define __MATHFUNCTION_H__

#include "cuda_runtime.h"

    // CUDA: use 16*16<512 threads per block
    const int CUDA_NUM_THREADS = 64;
    // CUDA: number of blocks for threads.
    inline int GET_BLOCKS(const int N)
    {
        return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
    }
    
    void process(float *src, float *dst, int old_w,int old_h, int new_w, int new_h);
#endif
