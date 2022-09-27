#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>

using std::cout;
using std::endl;

__global__ void add(int *dA)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    //dA[index] = 1;
    int sol = threadIdx.x + blockIdx.x;
    std::printf("%d %d" , index, sol);
}


int main(void)
{
    int N = 16;
    int *hA;     
    int *dA;
    int size = N * sizeof(int);
    cudaError_t cudaStatus;
    // Allocate space for device and host array a
    cudaMalloc((void **)&dA, size);
    hA = (int *)malloc(size);
    // Fill hA array on host
    for(int i = 0; i < N ; i++)
        hA[i] = 0;
    //Copy data from host to device
    cudaMemcpy(dA, hA, size, cudaMemcpyHostToDevice);
    // Launch add() kernel on GPU with 2 block and 8 threads.
    add<<<2, 8>>>(dA );
   // Copy result back to host
    cudaStatus = cudaMemcpy(hA, dA, size, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy returned error code %d after copying from kernel!\n", cudaStatus);
        return 0;
    }

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