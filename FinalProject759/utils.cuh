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

//Macro for checking cuda errors and rand errors following a cuda launch or api call

#define CHECK_CUDA(call) do { \
  cudaError_t status = call; \
  if( status != cudaSuccess ) { \
    fprintf(stderr, "CUDA Error at line %d in %s: %s\n", __LINE__, __FILE__, cudaGetErrorString(status)); \
    exit((int) status); \
  } \
} while(0)

#define CHECK_CURAND(call) do { \
  curandStatus_t status = call; \
  if( status != CURAND_STATUS_SUCCESS ) { \
    fprintf(stderr, "CURAND Error at line %d in %s: %d\n", __LINE__, __FILE__, status); \
    exit((int) status); \
  } \
} while(0)

static void dump_to_file(const char *name, int timestep, const double *data, int count)
{
  char buffer[256];
  sprintf(buffer, "%s-%d.bin", name, timestep);
  FILE *file = fopen(buffer, "wb");
  if( !file ) 
  {
    fprintf(stderr, "Error cannot open file %s\n", buffer);
    exit(1);
  }
  printf("> Debug info          : Writing %s to binary file %s\n", name, buffer);
  if( count != fwrite(data, sizeof(double), count, file) )
  {
    fprintf(stderr, "Error when dumping the binary values to %s\n", buffer);
    exit(1);
  }
  fclose(file);
}

#endif //PROFILE_H
