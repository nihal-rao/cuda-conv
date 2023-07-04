#include <bits/stdc++.h>
#include <cuda.h>
#include "conv_gpu.cuh"

__global__ void convGpuTiled(const float *in, float *out, int H_in, int W_in, int K, int TILE_SIZE)
{
    __shared__ float tile[32 * 32];
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    int H_out = H_in - K + 1;
    int W_out = W_in - K + 1;
    float sum = 0.0;

    if(row<H_in && col<W_in)
    {
        tile[threadIdx.y*32+threadIdx.x] = in[row*W_in+col];
    }
    __syncthreads();

    if(threadIdx.y<TILE_SIZE && threadIdx.x<TILE_SIZE && row < H_out && col < W_out)
    {   
        for (int i = 0; i < K; i++)
        {
            for (int j = 0; j < K; j++)
            {
                sum += d_mask[i * K + j] *tile[(i + threadIdx.y) * 32 + j + threadIdx.x];
            }
        }
        out[row*W_out+col]=sum;
    }
}
