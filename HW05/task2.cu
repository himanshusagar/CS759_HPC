#include "matmul.cuh"
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "profile.cuh"

#include <iostream>
#include <string>
#include <random>

using std::cout;
using std::endl;

template <typename T>
int perf_matmul(int N, int threads_per_block)
{

  T *A, *B, *C;
  size_t size = N * N * sizeof(T);
  cudaError_t cudaStatus;
  // Generate Random Values for kernel
  std::random_device entropy_source;
  std::mt19937 generator(entropy_source());
  std::uniform_real_distribution<T> dist(-1.0, 1.0);

  // Allocate Unified Memory -- accessible from CPU or GPU
  cudaMallocManaged(&A, size);
  cudaMallocManaged(&B, size);
  cudaMallocManaged(&C, size);

  // initialize A,B and C matrices on the host
  for (int i = 0; i < N * N; i++)
  {
    A[i] = 1;
    B[i] = 2;
    C[i] = 0;
  }

  matmul<T>(A, B, C, threads_per_block);
  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  // Free memory
  cudaFree(A);
  cudaFree(B);
  return 0;
}

int main(int argc, char *argv[])
{
  if (argc != 3)
  {
    cout << "Usage ./task1 N block_dim" << endl;
    return 0;
  }
  size_t N = std::stoi(argv[1]);
  size_t threads_per_block = std::stoi(argv[2]);
  perf_matmul<float>(N, threads_per_block);
  
}
