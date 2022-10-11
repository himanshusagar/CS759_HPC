#ifndef PROFILE_H
#define PROFILE_H

#include <iostream>
#include <chrono>
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

struct UnitGPUTime
{
    cudaEvent_t start;
    cudaEvent_t stop;
    UnitGPUTime() {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
     }
    float getTime()
    {
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        return ms;
    }
};

struct UnitCPUTime
{
    std::chrono::high_resolution_clock::time_point begin;
    UnitCPUTime() : begin(std::chrono::high_resolution_clock::now()) { }
    ~UnitCPUTime()
    {
        auto d = std::chrono::high_resolution_clock::now() - begin;
        float countValue = std::chrono::duration_cast<std::chrono::microseconds>(d).count();
        std::cout << countValue/1000 << std::endl;
    }
};

#endif //PROFILE_H
