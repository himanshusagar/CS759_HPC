#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "matmul.cuh"

#include <iostream>
#include <string> 
#include <random>

using std::cout;
using std::endl;

int main(void)
{
    int N = 10;
    int threads_per_block = 512;
    float *a, *b, *c;     
    float *a_d, *b_d, *c_d;
    int size = N * N * sizeof(float);
    cudaError_t cudaStatus;
    // Generate Random Values for kernel
    std::random_device entropy_source;
    std::mt19937 generator(entropy_source()); 
    std::uniform_int_distribution<int> dist(1,1000);
    
    // Allocate space for device and host array a
    cudaMalloc((void **)&a_d, size);
    cudaMalloc((void **)&b_d, size);
    cudaMalloc((void **)&c_d, size);
    
    a = (float *)malloc(size);
    b = (float *)malloc(size);
    c = (float *)malloc(size);
    
    // Fill a, b, c array on host
    for(int i = 0; i < N * N ; i++)
    {
        a[i] = i;
        b[i] = 1;
        c[i] = 0;
    }

    //Copy data from host to device
    cudaMemcpy(a_d, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b, size, cudaMemcpyHostToDevice);
    cudaMemcpy(c_d, c, size, cudaMemcpyHostToDevice);
    
    matmul(a_d , b_d , c_d, N , threads_per_block);
    // Copy result back to host
    cudaStatus = cudaMemcpy(c, c_d, size, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy returned error code %d after copying from kernel!\n", cudaStatus);
        return 0;
    }
    //Priting out device filled output
    for(int i = 0; i < N ; i++)
    {
        for(int j = 0; j < N ; j++)
        {
            cout << c[i * N + j] << " ";
        }
        cout << endl;
    }

    // Cleanup
    free(a);
    free(b);
    free(c);
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
    return 0;
}