#include "reduce.cuh"
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "profile.cuh"

#include <iostream>
#include <string> 
#include <random>

using std::cout;
using std::endl;

__global__ void reduce_kernel(float *g_idata, float *g_odata, unsigned int n)
{
    extern __shared__ int sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i   = blockIdx.x * ( blockDim.x * 2) + threadIdx.x;
    sdata[ tid ] = g_idata[ i ] + g_idata[ i + blockDim.x];
    g_odata[blockIdx.x] = 0;
    __syncthreads();
    for(unsigned int s = blockDim.x/2 ; s > 0 ; s >>= 1) 
    {
        if(tid < s) 
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if(tid == 0) 
        g_odata[blockIdx.x] = sdata[0];

}

__host__ void reduce(float **input, float **output, unsigned int N,
                     unsigned int threads_per_block)
{
    float* tmpIn = *input;
    float* tmpOut = *output;
    int out_N = N;
    
    bool shouldSwap = true;
    for(int iter_N = N ; iter_N > 1; )
    {
        double f_N = iter_N;
        double array_size_per_block = 2 * threads_per_block;
        int grid_size = ceil( f_N / array_size_per_block ); // grid_size X 1.
        if(iter_N == (int)N)
        {
            out_N = grid_size;
        }
        size_t shared_mem_size = ( threads_per_block * sizeof( float ) );
        //cout << "reduce " << iter_N << " " << array_size_per_block << " " <<  grid_size << "X" << threads_per_block << endl;

        // Call reduce_kernel
        reduce_kernel<<< grid_size, threads_per_block, shared_mem_size >>>( tmpIn, tmpOut , iter_N);
        //Mem Set extra folks as 0 so that they don't meddle in sum
        cudaMemset(tmpOut + grid_size , 0 , out_N - grid_size);
        cudaDeviceSynchronize();
        // Now grid size number of threads are left.
        iter_N = grid_size;
        //Swapping
        std::swap(tmpOut , tmpIn);
        shouldSwap = !shouldSwap;     
    }   
    
    if(shouldSwap)
    {
        *input = tmpOut;
        *output = tmpIn;
    }
}