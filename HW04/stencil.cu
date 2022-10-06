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
    float image_val;

    if(i >= n*n)
        return;

    int neg_R = (int)R * -1;
    int pos_R = R;
    int sq_n_1 = n * n - 1;
    // Per Block load R and Calculate indices of matrix relevant to block using blockDim.
    //HS : int mask_size = 2 * R + 1;
    __shared__ float sharedMask[257]; // should be mask size

    for(int j = 0 ; j < 2 * R + 1 ; j++ )
        sharedMask[j] = mask[j];

    int beginIndex = blockIdx.x * blockDim.x;
    int endIndex = beginIndex + blockDim.x - 1; // Highest Thread Index of Block.
    //BeginIndex cannot be OOR due to i.
    endIndex = endIndex >= sq_n_1 ? sq_n_1 : endIndex;
    int sharedImgSize = endIndex - beginIndex + 1; // Load Entire Row;

    __shared__ float sharedImg[ 1024 ];
    
    for(int k = 0; k < sharedImgSize ; k++)
    {
        sharedImg[k] = image[ k + beginIndex];
    }
    __syncthreads();

    for(int j = neg_R ; j <= pos_R ; j++ )
    {
        int sImgIndex = i + j - beginIndex;
        if( ( 0 <= (i + j) ) && ( (i + j) <= sq_n_1 ) )
            image_val = sharedImg[ sImgIndex ];
        else
            image_val = 1.0;
        output[i] += image_val * sharedMask[j + R];
    }

    std::printf(
        "blockIdx %d %d blockDim %d %d threadIdx %d %d output[%d] = %f \n"
        , blockIdx.x, blockIdx.y , blockDim.x , blockDim.y ,threadIdx.x , threadIdx.y , i,  output[i]);

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
    stencil_kernel<<<  grid_size, threads_per_block >>>(image, mask, output, n , R);
}