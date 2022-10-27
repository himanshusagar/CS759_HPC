#define CUB_STDERR // print CUDA runtime errors to console
#include <stdio.h>
#include <cub/util_allocator.cuh>
#include <cub/device/device_reduce.cuh>
#include "cub/util_debug.cuh"

#include <iostream>
#include <string>
#include <random>
#include <cmath>

#include "profile.cuh"

using namespace cub;
CachingDeviceAllocator  g_allocator(true);  // Caching allocator for device memory

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        std::cout << "Usage ./task_cub n " << std::endl;
        return 0;
    }
    size_t N = std::stoi(argv[1]);
    size_t size = N * sizeof(float);
    // Generate Random Values for kernel 
    std::random_device entropy_source;
    std::mt19937 generator(entropy_source());
    std::uniform_real_distribution<float> dist(-1, 1);

    // Set up host arrays
    float h_in[N];
    for (size_t i = 0; i < N; i++)
    {
        h_in[i] = dist(generator);
    }

    // Set up device arrays
    float* d_in = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)& d_in, size ));
    // Initialize device input
    CubDebugExit(cudaMemcpy(d_in, h_in, size , cudaMemcpyHostToDevice));
    // Setup device output array
    float* d_sum = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)& d_sum, sizeof(float) * 1));
    // Request and allocate temporary storage
    void* d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    CubDebugExit(DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_sum, N));
    CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));

    // Do the actual reduce operation'
    float time_val;
    {
        UnitGPUTime gTime;
        DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_sum, N);
        time_val = gTime.getTime();
    }
    float gpu_sum;
    CubDebugExit(cudaMemcpy(&gpu_sum, d_sum, sizeof(int) * 1, cudaMemcpyDeviceToHost));
    
    //Print out sum and time taken
    std::cout << gpu_sum << std::endl << time_val << std::endl;
    
    // Cleanup
    if (d_in) CubDebugExit(g_allocator.DeviceFree(d_in));
    if (d_sum) CubDebugExit(g_allocator.DeviceFree(d_sum));
    if (d_temp_storage) CubDebugExit(g_allocator.DeviceFree(d_temp_storage));
    return 0;
}
