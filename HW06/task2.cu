#include "scan.cuh"
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
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
    cout << "Usage ./task1 n threads_per_block" << endl;
    return 0;
  }
  size_t N = std::stoi(argv[1]);
  size_t threads_per_block = std::stoi(argv[2]);

  // Generate Random Values for kernel
  std::random_device entropy_source;
  std::mt19937 generator(entropy_source());
  std::uniform_real_distribution<float> dist(-1, 1);

  float *input, *output;
  size_t size = N * sizeof(float);
  // Allocate Unified Memory -- accessible from CPU or GPU
  cudaMallocManaged(&input, size);
  cudaCheckError();
  cudaMallocManaged(&output, size);
  cudaCheckError();    

  // initialize A,B and C matrices on the host and gpu
  for (size_t i = 0; i < N; i++)
  {
    input[i] = 1;
    output[i] = 0;
  }

  float time_val;
  {
    UnitGPUTime g;
    scan(input, output , N , threads_per_block);
    time_val = g.getTime();
  }

  std::cout << std::log2(N) << "," <<time_val << endl;

  // free unified arrays.
  cudaFree(input);
  cudaFree(output);
  return 0;
}
