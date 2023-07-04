#include <bits/stdc++.h>
#include <cuda.h>
#include "conv_gpu.cuh"

__global__ void convGpuNaive(const float *in, float *out, int H_in, int W_in, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int H_out = H_in - K + 1;
    int W_out = W_in - K + 1;
    float sum = 0.0;

    if (row < H_out && col < W_out)
    {
        for (int i = 0; i < K; i++)
        {
            for (int j = 0; j < K; j++)
            {
                sum += d_mask[i * K + j] *in[(i+row) * W_in + j+col];
            }
        }
        out[row*W_out+col]=sum;
    }
}
