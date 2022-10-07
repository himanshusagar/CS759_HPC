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
    int globalIndex = blockIdx.x * blockDim.x + threadIdx.x;

    extern __shared__ float g_shared_mem[];

    // Prepare variables for loops.
    int pos_R = R;
    int neg_R = pos_R * -1;
    int N = n;
    int N_1 = N - 1;


    //Load portion of shared memory for mask
    float *sharedMask = g_shared_mem;
    int sharedMaskSize = 2 * pos_R + 1;
    //We're given that 
    //block size >= 2 * R + 1;
    //For Mask, local Index is global Index.
    int localMaskIndex = threadIdx.x;

    if(localMaskIndex < sharedMaskSize)
    {
        sharedMask[localMaskIndex] = mask[localMaskIndex];
    }

    //Load portion of shared memory for image
    float* sharedImg = g_shared_mem + sharedMaskSize; 
    int localBeginIndex = neg_R;
    int localEndIndex = blockDim.x + pos_R;
    int sharedImgSize = localEndIndex - localBeginIndex + 1;
    int DIFF = N_1 -  blockIdx.x * blockDim.x + 1;
    int BLK_OFFSET = DIFF > blockDim.x ? blockDim.x : DIFF; // Need

    //Fill My Location;
    sharedImg[threadIdx.x + pos_R] = globalIndex <= N_1 ? image[globalIndex] : 1.0;
    if(threadIdx.x < pos_R)
    {
        //Fill Left Size
        if( globalIndex - pos_R  >= 0)
            sharedImg[threadIdx.x] = image[globalIndex - pos_R];
        else
            sharedImg[threadIdx.x] = 1.0;
        //Fill Right Side
        // Right Side offset can be blockDim.x OR it'd be max index.
        
        if( globalIndex + BLK_OFFSET <= N_1)
            sharedImg[pos_R + BLK_OFFSET + threadIdx.x ] = image[globalIndex + BLK_OFFSET];
        else
            sharedImg[pos_R + BLK_OFFSET + threadIdx.x ] = 1.0;
    }

    if(globalIndex > N_1)
        return;

    __syncthreads(); // Mask and SharedImg are filled now.

 

    float *outputSharedMem = g_shared_mem + sharedMaskSize + sharedImgSize;
    outputSharedMem[ threadIdx.x ] = 0;
    for(int j = neg_R ; j <= pos_R ; j++ )
    {
        //Local's -R is 
        int sImgIndex = threadIdx.x + pos_R + j;
        outputSharedMem[ threadIdx.x ] += sharedImg[ sImgIndex ] * sharedMask[j + R];   
    }

    output[globalIndex] = outputSharedMem[ threadIdx.x ];

    // std::printf(
    //     "blockIdx %d blockDim %d threadIdx %d output[%d] = %f , beginIndex %d , endIndex %d ,  %f %f %f %f DIFF %d BLK_OFFSET %d MASK %f\n"
    //     , blockIdx.x , blockDim.x, threadIdx.x , globalIndex,  output[globalIndex],  R + threadIdx.x  ,  R + blockDim.x + threadIdx.x 
    //     , sharedImg[ pos_R + threadIdx.x ], 
    //     image[ blockIdx.x * blockDim.x + threadIdx.x ] ,
    //     sharedImg[ pos_R + threadIdx.x - 1] ,
    //     sharedImg[ pos_R + threadIdx.x + 1 ],
    //     DIFF,
    //     BLK_OFFSET,
    //     sharedMask[threadIdx.x]
    //     );

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
    int grid_size = ceil( f_n / (float)threads_per_block );
    //Mask Size
    //Image Size
    size_t shared_img_size = threads_per_block + R + R + 1;
    size_t shared_out_size = threads_per_block;
    size_t shared_mem_size = ( (2 * R + 1) + shared_img_size + shared_out_size ) * sizeof(float);
    stencil_kernel<<<  grid_size, threads_per_block, shared_mem_size >>>(image, mask, output, n , R);
}