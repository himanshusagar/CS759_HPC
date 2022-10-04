#include "stencil.cuh"
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <string> 
#include <random>

using std::cout;
using std::endl;


__global__ void stencil_kernel(const float* image, const float* mask, float* output, unsigned int n, unsigned int R)
{
    //Matrix is n * n.
    int i = blockIdx.x * blockDim.x + threadIdx.x, image_val;

    if(i >= n*n)
        return;

    int row = i / n;
    int col = i % n;

    for(int j = -R ; j <= R ; j++ )
    {
        if( (i + j) < 0 || (i + j) >= n*n )
            image_val = 1;
        else
            image_val = image[i + j];
        output[i] += image_val * mask[j + R];
    }

    std::printf(
        "blockIdx %d %d blockDim %d %d threadIdx %d %d output[%d][%d] = %f \n"
        , blockIdx.x, blockIdx.y , blockDim.x , blockDim.y ,threadIdx.x , threadIdx.y , row, col, output[i]);

}

__host__ void stencil(const float* image,
    const float* mask,
    float* output,
    unsigned int n,
    unsigned int R,
    unsigned int threads_per_block)
{
    // Launch simple kernel on GPU with 2 block and 8 threads.
    float f_n = n;
    int grid_size = ceil(  f_n * ( f_n / (float)threads_per_block ) );
    stencil_kernel<<<  grid_size, threads_per_block >>>(image, mask, output, n , R);
}