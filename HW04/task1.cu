#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "matmul.cuh"
#include "profile.cuh"

#include <iostream>
#include <string> 
#include <random>

using std::cout;
using std::endl;

int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        cout << "Usage ./task1 n threads per block" << endl;
        return 0;
    }
    size_t N = std::stoi(argv[1]);
    size_t threads_per_block = std::stoi(argv[2]);

    float *a, *b, *c;     
    float *d_a, *d_b, *d_c;
    size_t size = N * N * sizeof(float);
    cudaError_t cudaStatus;
    // Generate Random Values for kernel
    std::random_device entropy_source;
    std::mt19937 generator(entropy_source()); 
    std::uniform_real_distribution<float> dist(-1.0,1.0);
    
    // Allocate space for device and host array a
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);
    
    a = (float *)malloc(size);
    b = (float *)malloc(size);
    c = (float *)malloc(size);
    
    // Fill a, b, c array on host
    for(size_t i = 0; i < N * N ; i++)
    {
        a[i] = dist(generator);
        b[i] = dist(generator);
        c[i] = 0;
    }

    //Copy data from host to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, c, size, cudaMemcpyHostToDevice);
    float time_taken = 0;
    {
        UnitGPUTime g;
        matmul(d_a , d_b , d_c, N , threads_per_block);
        time_taken = g.getTime();
    }
    // Copy result back to host
    cudaStatus = cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy returned error code %d after copying from kernel!\n", cudaStatus);
        return 0;
    }

    cout << c[N * N - 1] << endl << time_taken << endl;

    // Cleanup
    free(a);
    free(b);
    free(c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;
}