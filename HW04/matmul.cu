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

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    //C[index] is what we're solving for.
    if(row >= n || col >= n)
        return;
    //std::printf(
        //"blockIdx %d %d blockDim %d %d threadIdx %d %d A[%d][%d] = %f \n"
        //, blockIdx.x, blockIdx.y , blockDim.x , blockDim.y ,threadIdx.x , threadIdx.y , row, col, A[ row*n + col]);

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
    int grid_size = ceil( (float)(n) / (float)block_size );
    dim3 dim_grid( grid_size , grid_size , 1);
    dim3 dim_block( block_size, block_size, 1);

    matmul_kernel<<<  dim_grid, dim_block >>>(A, B, C, n);
}