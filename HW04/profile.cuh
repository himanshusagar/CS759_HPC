#ifndef PROFILE_H
#define PROFILE_H

#include <iostream>
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

#endif //PROFILE_H
