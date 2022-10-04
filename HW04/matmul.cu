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
    int row = index / n;
    int col = index % n;
    //C[index] is what we're solving for.
    if(index >= n*n)
        return;
    // std::printf(
    //     "blockIdx %d %d blockDim %d %d threadIdx %d %d A[%d][%d] = %f \n"
    //     , blockIdx.x, blockIdx.y , blockDim.x , blockDim.y ,threadIdx.x , threadIdx.y , row, col, A[ row*n + col]);

    int i_index, j_index;
    for(int k = 0; k < n; k++)
    {
        i_index = row * n + k;
        j_index = k * n + col;
        C[row * n + col] += A[i_index] * B[j_index];
    }
}

void matmul(const float* A, const float* B, float* C, size_t n, unsigned int block_size)
{
    // Launch simple kernel on GPU with 2 block and 8 threads.
    float f_n = n;
    int grid_size = ceil(  f_n * ( f_n / (float)block_size ) );
    matmul_kernel<<<  grid_size, block_size >>>(A, B, C, n);
}