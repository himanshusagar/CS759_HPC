#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>

using std::cout;
using std::endl;

__global__ void add(int *a, int *c)
{
    c[threadIdx.x] = 1;
    for( int i = 1 ; i <= a[threadIdx.x]; i++)
    {
        c[threadIdx.x] = c[threadIdx.x] * (i);
    }
    std::printf("%d!=%d\n", a[threadIdx.x], c[threadIdx.x] );
}

int main(void)
{
    int N = 16;
    int *a,  *c;       
    int *d_a,  *d_c;   
    int size = N * sizeof(int);
    cudaError_t cudaStatus;
    // Allocate space for device copies of a, b, c
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_c, size);
    // Allocate space for host copies of a, b, c and setup input values
    a = (int *)malloc(size);
    c = (int *)malloc(size);

    // Fill A array on host
    for(int i = 0; i < N ; i++)
        a[i] = i+1;
    //Copy data from host to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    // Launch add() kernel on GPU with 1 block and N threads.
    add<<<1, N>>>(d_a, d_c);
    // Synchronize and see if we were successful.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        return 0;
    }

    // Copy result back to host
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
    // Cleanup
    free(a);
    free(c);
    cudaFree(d_a);
    cudaFree(d_c);
    return 0;
}