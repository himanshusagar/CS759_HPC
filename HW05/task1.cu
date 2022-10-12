#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "reduce.cuh"
#include "profile.cuh"

#include <iostream>
#include <string> 
#include <random>

using std::cout;
using std::endl;


void printX(float *val, int N)
{
    std::cout << "Array : " << std::endl;
    for(int i = 0; i < N ; i++)
    {
        std::cout << val[i] << " ";
    }
    std::cout << std::endl;
}

int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        cout << "Usage ./task1 N threads per block" << endl;
        return 0;
    }
    //Prepare variable for calculations
    size_t N = std::stoi(argv[1]);
    size_t threads_per_block = std::stoi(argv[2]);

    float *input, *output;     
    float *d_input, *d_output;
    size_t size = N * sizeof(float);
    
    // Generate Random Values for kernel
    std::random_device entropy_source;
    std::mt19937 generator(entropy_source()); 
    std::uniform_real_distribution<float> dist(-1.0,1.0);
    
    // Allocate space for device and host array a
    cudaMalloc((void **)&d_input, size);
    cudaCheckError();
    cudaMalloc((void **)&d_output, size);
    cudaCheckError();
    
    input = (float *)malloc(size);
    output = (float *)malloc(size);
    
    // Fill a, b, c array on host
    for(size_t i = 0; i < N ; i++)
    {
        input[i] = i + 1;
        output[i] = 0;
    }

    //Copy data from host to device
    cudaMemcpy(d_input, input, size, cudaMemcpyHostToDevice);
    cudaCheckError();
    cudaMemcpy(d_output, output, size, cudaMemcpyHostToDevice);
    cudaCheckError();

    float time_taken = 0;
    {
        UnitGPUTime g;
        reduce(&d_input, &d_output, N, threads_per_block);
        time_taken = g.getTime();
    }
    // Copy result back to host
    cudaMemcpy(output, d_output, size, cudaMemcpyDeviceToHost);
    cudaCheckError();
    cudaMemcpy(input, d_input, size, cudaMemcpyDeviceToHost);
    cudaCheckError();
    //Print last element and time taken.
    // printX(input, N);
    // printX(output, N);
    
    cout << std::log2(N) << "," << time_taken << endl;

    // Cleanup
    free(input);
    free(output);
    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}