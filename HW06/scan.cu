
#include "scan.cuh"
#include "profile.cuh"

using std::cout;
using std::endl;


__global__ void merge_results_kernel(float* g_odata, const float *g_addSum, int N, int grid_size, int threads_per_block)
{
    int tid = threadIdx.x;
    if(tid >= N)
        return;
    int row = threads_per_block;
    // Merging results is just summing over everything.
    for(int i = 0 ; i < row ; i++)
    {
        int index = tid*row + i;
        if( index < N)
        {
            g_odata[ index ] = g_odata[ index ] + g_addSum[ tid ];
        }
    }
}
// Copied from lecture slides but changed for running all blocks.
__global__ void scan_iter_kernel(const float* g_idata , float* g_odata, int N)
{   
    extern volatile __shared__ float sharedMem[];
    int g_index = blockDim.x * blockIdx.x + threadIdx.x;
    int tid = threadIdx.x;
    int pout = 0, pin = 1;

    int bDim = blockDim.x;
    sharedMem[tid] = 0;
    if(g_index < N)
    {
        sharedMem[tid] = g_idata[g_index];
    }
    __syncthreads();
    for( int offset = 1; offset < bDim; offset *= 2 ) 
    {
        pout = 1 - pout;
        pin = 1 - pin;

        if(tid >= offset) // See if we need computation here.
        {
            sharedMem[ pout*bDim + tid ] = sharedMem[pin*bDim + tid] + sharedMem[ pin*bDim + tid - offset]; 
        }
        else
        {
            sharedMem[ pout*bDim + tid] = sharedMem[ pin*bDim + tid];
        }
        
        __syncthreads();
    }

    if(g_index < N) // If in range then set output.
    {
        g_odata[g_index] = sharedMem[pout*bDim + tid];
    }
}


__host__ int scan_iter(const float* input, float* output, unsigned int N, unsigned int block_dim)
{
    // Compute paramters to launch kernel
    float f_N = N;
    float f_block_dim = block_dim;
    size_t grid_size = ceil(f_N / f_block_dim);
    size_t shared_mem_size = 2 * block_dim * sizeof(float); 
    // Launch scan kernel
    scan_iter_kernel<<< grid_size, block_dim , shared_mem_size >>>(input, output, N);
    cudaDeviceSynchronize();
    cudaCheckError();  
    return grid_size;
}

__host__ void scan(const float* input, float* output, unsigned int n, unsigned int threads_per_block)
{

    int N = n;
    int new_N;
    float *block_ip , *block_op;
    {
        // If only block suffices, then we're done.
        new_N = scan_iter(input , output , N , threads_per_block);
        if(new_N <= 1)
            return;
        size_t new_size = new_N * sizeof( float );
        cudaMallocManaged(&block_ip, new_size);
        cudaCheckError();  
        cudaMallocManaged(&block_op, new_size);
        cudaCheckError();  
        block_ip[0] = 0;
        // Generate middle input array for computation.
        for(int i = 1 , output_index = threads_per_block ; i < new_N ; i++)
        {
            output_index = std::min(N , output_index);
            block_ip[i] = output[output_index - 1];
            output_index += threads_per_block;
        }
    }
    // Run one more iteration of scan to accumulate for all blocks
    scan_iter(block_ip , block_op , new_N , threads_per_block);
    cudaCheckError();
    // Merge results by summing over entire range for final output.
    merge_results_kernel<<< 1 , threads_per_block >>>(output, block_op, N, new_N, threads_per_block);
    // Sync and check for errors.
    cudaDeviceSynchronize();
    cudaCheckError();
}