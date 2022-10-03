#include "matmul.cuh"
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <string> 
#include <random>

using std::cout;
using std::endl;

__global__ void matmul_kernel(const float* A, const float* B, float* C, size_t n)
{
    //Matrix is n * n.
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    //C[index] is what we're solving for.
    if( index < (n * n) )
    {
        float f = 0;
        for(int k = 0; k < n; k++)
        {
            int i_index = blockIdx.x * blockDim.x + k;
            int j_index = blockIdx.x * k + threadIdx.x;
            f += A[i_index] * B[j_index];
        }
        C[index] = f;    
    }
}

void matmul(const float* A, const float* B, float* C, size_t n, unsigned int threads_per_block)
{
    // Launch simple kernel on GPU with 2 block and 8 threads.
    unsigned int block_count =  (n * n + threads_per_block - 1) / threads_per_block;
    matmul_kernel<<<  block_count, threads_per_block >>>(A, B, C, n);
    // Synchronize and see if we were successful.
    cudaError_t cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) 
    {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        return;
    }
}