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
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i >= n*n)
        return;
    extern __shared__ float g_shared_mem[];

    // Prepare variables for loops.
    int neg_R = (int)R * -1;
    int pos_R = R;
    int sq_n_1 = n * n - 1;

    //Load portion of shared memory for mask
    float *sharedMask = g_shared_mem;
    int sharedMaskSize = 2 * R + 1;
    for(int j = 0 ; j < sharedMaskSize ; j++ )
        sharedMask[j] = mask[j];

    //Load portion of shared memory for image
    float* sharedImg = g_shared_mem + sharedMaskSize;    
    int beginIndex = blockIdx.x * blockDim.x - R;
    int endIndex = beginIndex + blockDim.x + R; // Highest Thread Index of Block.
    for(int l = beginIndex , k = 0 ; l <= endIndex ; l++ , k++)
    {
        if( ( 0 <= l ) && ( l <= sq_n_1 ) )
            sharedImg[k] = image[l];
        else
            sharedImg[k] = 1;
    }

    __syncthreads();

    for(int j = neg_R ; j <= pos_R ; j++ )
    {
        int sImgIndex = i + j - beginIndex;
        output[i] += sharedImg[ sImgIndex ] * sharedMask[j + R];
    }

    __syncthreads();
    std::printf(
        "blockIdx %d blockDim %d threadIdx %d output[%d] = %f , beginIndex %d , endIndex %d \n"
        , blockIdx.x , blockDim.x, threadIdx.x , i,  output[i], beginIndex, endIndex );

}

__host__ void stencil(const float* image,
    const float* mask,
    float* output,
    unsigned int n,
    unsigned int R,
    unsigned int threads_per_block)
{
    // Launch simple kernel on GPU with 2 block and 8 threads.
    double f_n = n;
    int grid_size = ceil(  f_n * ( f_n / (float)threads_per_block ) );
    cout << grid_size << " X " << threads_per_block << endl;
    //Mask Size
    //Image Size
    size_t shared_img_size = threads_per_block + R + R;
    size_t shared_out_size = threads_per_block;
    size_t shared_mem_size = ( (2 * R + 1) + shared_img_size + shared_out_size ) * sizeof(float);
    stencil_kernel<<<  grid_size, threads_per_block, shared_mem_size >>>(image, mask, output, n , R);
}