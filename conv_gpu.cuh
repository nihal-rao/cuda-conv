#ifndef __CONV_GPU_H
#define __CONV_GPU_H
__global__ void convGpu(const float *in, float *out, int H_in, int W_in, int K);
extern __constant__ float d_mask[256];
extern __constant__ float debug;
#endif