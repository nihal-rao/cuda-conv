#include <bits/stdc++.h>
#include <cuda.h>
#include "conv_gpu.cuh"

__global__ void convGpu(const float *in, float *out, int H_in, int W_in, int K)
{
    __shared__ float tile[32 * 32];
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int H_out = H_in - K + 1;
    int W_out = W_in - K + 1;
    float sum = 0.0;
    int row_lim = min(H_out, blockDim.y*blockIdx.y);
    int col_lim = min(W_out, blockDim.x*blockIdx.x);

    if (row < H_out && col < W_out)
    {
        tile[threadIdx.y * blockDim.x + threadIdx.x] = in[row * W_in + col];
        __syncthreads();
        for (int i = 0; i < K; i++)
        {
            for (int j = 0; j < K; j++)
            {
                if (i + row >= row_lim || j + col >= col_lim)
                {
                    sum += d_mask[i * K + j] *in[(i+row) * W_in + j+col];
                }
                else
                {
                    sum += d_mask[i * K + j] *tile[(i + threadIdx.y) * blockDim.x + j + threadIdx.x]; //
                }
            }
        }
        out[row*W_out+col]=sum;
    }
}
