#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "vscale.cuh"
#include "profile.cuh"

#include <iostream>
#include <string> 
#include <random>

using std::cout;
using std::endl;


int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        cout << "Usage ./task1 N" << endl;
        return 0;
    }
    int N = std::stoi(argv[1]);
    int THREAD_COUNT = 512;

    // Generate Random Values using real dist
    std::random_device entropy_source;
	std::mt19937 generator(entropy_source()); 
	std::uniform_real_distribution<float> dist_10(-10.0, 10.0);
    std::uniform_real_distribution<float> dist_01(0.0, 1.0);

    float *a,  *b;       
    float *d_a,  *d_b;   
    size_t size = N * sizeof(float);
    cudaError_t cudaStatus;
    // Allocate space for device copies of a, b, c
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    // Allocate space for host copies of a, b, c and setup input values
    a = (float *)malloc(size);
    b = (float *)malloc(size);

    // Fill A array on host
    for(int i = 0; i < N ; i++)
    {
        a[i] = dist_10(generator);
        b[i] = dist_01(generator);        
    }
    //Copy data from host to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
    
    float f_N = N;
    int block_count = ceil( f_N / THREAD_COUNT);
    // Launch vscale() kernel on GPU with 1 block and N threads.
    {
        UnitGPUTime g;
        vscale<<< block_count, THREAD_COUNT >>>(d_a, d_b, N);
    }
    // Copy result back to host
    cudaStatus = cudaMemcpy(b, d_b, size, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy returned error code %d after copying from kernel!\n", cudaStatus);
        return 0;
    }

    //Priting out device filled output
    cout << b[0] << endl << b[N-1] << endl;
    
    //Priting out device filled output
    // for(int i=0; i < N ; i++)
    // {
    //     cout << b[i] << " ";
    // }cout<<endl;
    // Cleanup
    free(a);
    free(b);
    cudaFree(d_a);
    cudaFree(d_b);
    return 0;
}