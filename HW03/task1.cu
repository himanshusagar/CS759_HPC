#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>

using std::cout;
using std::endl;

__global__ void add(int *a, int *c)
{
    c[blockIdx.x] = 1.0;
    for( int i = 1 ; i <= a[blockIdx.x]; i++)
    {
        c[blockIdx.x] = c[blockIdx.x] * (i);
    }
    std::printf("%d!=%d\n" , a[blockIdx.x] , c[blockIdx.x]);
}

int main(void)
{
    int N = 8;
    int *a,  *c;       // host copies of a, b, c
    int *d_a,  *d_c;   // device copies of a, b, c
    int size = N * sizeof(int);
    cudaError_t cudaStatus;
    // Alloc space for device copies of a, b, c
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_c, size);
    // Alloc space for host copies of a, b, c and setup input values
    a = (int *)malloc(size);
    for(int i = 0; i < N ; i++)
    {
        a[i] = i+1;
    }
    c = (int *)malloc(size);

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    // Launch add() kernel on GPU with N blocks
    add<<<N, 1>>>(d_a, d_c);

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        return 0;
    }

    // Copy result back to host
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
    for(int i = 0; i < N ; i++)
    {
        cout << a[i] << " " << c[i] << endl;
        fflush(stdout);
    }
    // Cleanup
    free(a);
    free(c);
    cudaFree(d_a);
    cudaFree(d_c);
    return 0;
}