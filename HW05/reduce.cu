#include "reduce.cuh"
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

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
    double f_N = N;
    unsigned int array_size_per_block = 2 * threads_per_block;
    int grid_size = ceil( f_N / array_size_per_block ); // grid_size X 1.
    size_t shared_mem_size = ( threads_per_block * sizeof( float ) );
    cout << "reduce " << N << " " << threads_per_block << " " <<  grid_size << " " << shared_mem_size << endl;

    if(grid_size < 1)
        return;
   
    // Call Kernel
    reduce_kernel<<<  grid_size, threads_per_block, shared_mem_size >>>(*input, *output, N);
    // Now grid size number of threads are left.
    // Now output is input
   // reduce(output , input, grid_size / array_size_per_block , threads_per_block);
}