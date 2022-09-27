#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>

using std::cout;
using std::endl;

__global__ void add(int *a)
{
    int sol = 1;
    for( int i = 1 ; i <= a[threadIdx.x]; i++)
    {
        sol = sol * (i);
    }
    std::printf("%d!=%d\n", a[threadIdx.x], sol);
}

int main(void)
{
    int N = 8;
    int *a;     
    int *d_a;
    int size = N * sizeof(int);
    cudaError_t cudaStatus;
    // Allocate space for device and host array a
    cudaMalloc((void **)&d_a, size);
    a = (int *)malloc(size);
    // Fill A array on host
    for(int i = 0; i < N ; i++)
        a[i] = i+1;
    //Copy data from host to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    // Launch add() kernel on GPU with 1 block and N threads.
    add<<<1, N>>>(d_a);
    // Synchronize and see if we were successful.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        return 0;
    }
    // Cleanup
    free(a);
    cudaFree(d_a);
    return 0;
}