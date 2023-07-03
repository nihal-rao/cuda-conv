#ifndef __CONV_H
#define __CONV_H
void convCpu( const float* __restrict in, float* __restrict out, const float* __restrict mask, int H_in, int W_in, int K);
#endif