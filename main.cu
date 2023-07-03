#include<bits/stdc++.h>
#include "conv.h"
#include "conv_gpu.cuh"
#include "unroll.cuh"
#include <cuda.h>
#include <cublas_v2.h>

#define THRESHOLD (0.001)

using std::cout;
using std::endl;
using std::chrono::steady_clock;

__constant__ float d_mask[256];
__constant__ float debug;

void check_result(const float* w_ref, const float* w_opt, int h, int w) 
{
  float maxdiff = 0.0, this_diff = 0.0;
  int numdiffs = 0;

  for (int i = 0; i < h; i++) {
    for (int j = 0; j < w; j++) {
        this_diff = w_ref[i*w+j] - w_opt[i*w+j];
        if (std::fabs(this_diff) > THRESHOLD) {
          numdiffs++;
          if (this_diff > maxdiff) {
            maxdiff = this_diff;
          }
        }
      }
    }

  if (numdiffs > 0) {
    cout << numdiffs << " Diffs found over THRESHOLD " << THRESHOLD << "; Max Diff = " << maxdiff
         << endl;
  } else {
    cout << "No differences found between base and test versions\n";
  }
}

void display_result(const float* w_opt, int h, int w) 
{

  for (int i = 0; i < h; i++) {
    for (int j = 0; j < w; j++) {
        cout<<w_opt[i*w+j]<<"  ";
      }
    cout<<endl;
    }
}

int main(int argc, char* argv[])
{
    int H_in = std::strtol(argv[1], nullptr, 0);
    int W_in = std::strtol(argv[2], nullptr, 0);
    int K = std::strtol(argv[3], nullptr, 0);

    int H_out = H_in-K+1;
    int W_out = W_in-K+1;
    float * in, *out, *mask, *h_out_gpu;
    in = (float*)malloc(sizeof(float)*H_in*W_in);
    out = (float*)malloc(sizeof(float)*H_out*W_out);
    h_out_gpu = (float*)malloc(sizeof(float)*H_out*W_out);
    mask = (float*)malloc(sizeof(float)*K*K);

    for(int i=0;i<H_in;i++)
    {
        for(int j=0;j<W_in;j++)
        {
            in[i*W_in+j]=rand()%256;
            // in[i*W_in+j]=1;
        }
    }

    for(int i=0;i<H_out;i++)
    {
        for(int j=0;j<W_out;j++)
        {
            out[i*W_out+j]=0.0;
        }
    }
    
    for(int i=0;i<K;i++)
    {
        for(int j=0;j<K;j++)
        {
        float scale = rand() / (float) RAND_MAX;
        mask[i*K+j]= -1 + scale * (2);
        // mask[i*K+j]=rand()%3;
        // mask[i*K+j]=1;
        }
    }

    auto start_cpu = steady_clock::now();
    convCpu(in, out, mask, H_in, W_in, K);
    auto end_cpu = steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_cpu - start_cpu).count();
    cout<<" CPU time is "<<duration<<" ms"<<endl;
    // cout<<" CPU result is "<<endl;
    // display_result(out, H_out, W_out);
    // display_result(in, H_in, W_in);

    float *d_in, *d_out;
    cudaMalloc(&d_in, sizeof(float)*H_in*W_in);
    cudaMalloc(&d_out, sizeof(double)*H_out*W_out);
    cudaMemcpy(d_in, in, sizeof(float)*H_in*W_in, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_mask, mask, sizeof(float)*K*K);

    int block_size_h = 32;
    int block_size_w = 32;
    
    dim3 threadsPerBlock(block_size_w, block_size_h);
    dim3 numBlocks((int)ceil(W_in/(float)block_size_w), (int)ceil(H_in/(float)block_size_h));

    cudaEvent_t start, stop;
    float gpu_time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    convGpu<<<numBlocks, threadsPerBlock>>>(d_in, d_out, H_in, W_in, K);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    
    cudaEventElapsedTime(&gpu_time, start, stop);
    // cudaEventDestroy(start);
    // cudaEventDestroy(stop);

    cout << "GPU Time is "<<gpu_time << " ms" << endl;

    cudaMemcpy(h_out_gpu, d_out, sizeof(float)*H_out*W_out, cudaMemcpyDeviceToHost);
    cout<<" GPU result is "<<endl;
    // display_result(h_out_gpu, H_out, W_out);
    check_result(out, h_out_gpu, H_out, W_out);

    cudaFree(d_in);
    cudaFree(d_out);
    
    free(in);
    free(out);
    free(mask);
    free(h_out_gpu)
}