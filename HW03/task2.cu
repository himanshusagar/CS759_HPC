#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>

using std::cout;
using std::endl;

__global__ void simple_kernel(int *dA, int a)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int sol = a * threadIdx.x + blockIdx.x;
    dA[index] = sol;
}


int main(void)
{
    int N = 16;
    int *hA;     
    int *dA;
    int size = N * sizeof(int);
    cudaError_t cudaStatus;
    // Generate Random Values for kernel
    std::random_device entropy_source;
    std::mt19937 generator(entropy_source()); 
    std::uniform_int_distribution<int> dist(1,1000);
    int random_num = dist(generator); 

    // Allocate space for device and host array a
    cudaMalloc((void **)&dA, size);
    hA = (int *)malloc(size);
    
    // Fill hA array on host
    for(int i = 0; i < N ; i++)
        hA[i] = 0;
    //Copy data from host to device
    cudaMemcpy(dA, hA, size, cudaMemcpyHostToDevice);
    // Launch simple kernel on GPU with 2 block and 8 threads.
    simple_kernel<<<2, 8>>>(dA , random_num);

   // Copy result back to host
    cudaStatus = cudaMemcpy(hA, dA, size, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy returned error code %d after copying from kernel!\n", cudaStatus);
        return 0;
    }
    //Priting out device filled output
    for(int i=0; i < N ; i++)
    {
        cout << hA[i] << " ";
    }
    cout << endl;
    // Cleanup
    free(hA);
    cudaFree(dA);
    return 0;
}