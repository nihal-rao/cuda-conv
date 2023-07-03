#include<bits/stdc++.h>
#include "conv.h"

void convCpu( const float* __restrict in, float* __restrict out, const float* __restrict mask, int H_in, int W_in, int K)
{
    int H_out = H_in-K+1;
    int W_out = W_in -K+1;
    for(int i=0;i<H_out;i++)
    {
        for(int j=0;j<W_out;j++)
        {
            for(int p=0;p<K;p++)
            {
                for(int q=0;q<K;q++)
                {
                    out[i*W_out+j]+=in[(i+p)*W_in+j+q]*mask[p*K+q];
                }
            }
        }
    }
}