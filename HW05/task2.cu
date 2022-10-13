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
int perf_matmul(int N, int block_dim)
{

  T *A, *B, *C;
  size_t size = N * N * sizeof(T);
  // Generate Random Values for kernel
  std::random_device entropy_source;
  std::mt19937 generator(entropy_source());
  std::uniform_real_distribution<T> dist(-1.0, 1.0);

  // Allocate Unified Memory -- accessible from CPU or GPU
  cudaMallocManaged(&A, size);
  cudaCheckError();
  cudaMallocManaged(&B, size);
  cudaCheckError();
  cudaMallocManaged(&C, size);
  cudaCheckError();

  // initialize A,B and C matrices on the host
  for (int i = 0; i < N * N; i++)
  {
    A[i] = 1;
    B[i] = 2;
    C[i] = 0;
  }

 // if (std::is_same<T, int>::value)
 // {
    matmul_2(A, B, C, N, block_dim);
    cudaCheckError();
  // }
  // if (std::is_same<T, float>::value)
  // {
  //   matmul_2(A, B, C, threads_per_block);
  // }
  // if (std::is_same<T, double>::value)
  // {
  //   matmul_3(A, B, C, threads_per_block);
  // }
  
  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
  cudaCheckError();

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
  size_t block_dim = std::stoi(argv[2]);
  perf_matmul<float>(N, block_dim);
  
}
