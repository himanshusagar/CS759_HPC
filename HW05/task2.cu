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

float *A_gen;
float *B_gen;

template <typename T>
int perf_matmul(size_t N, size_t block_dim)
{
  T *A, *B, *C;
  size_t size = N * N * sizeof(T);
  // Allocate Unified Memory -- accessible from CPU or GPU
  cudaMallocManaged(&A, size);
  cudaCheckError();
  cudaMallocManaged(&B, size);
  cudaCheckError();
  cudaMallocManaged(&C, size);
  cudaCheckError();

  // initialize A,B and C matrices on the host and gpu
  for (size_t i = 0; i < N * N; i++)
  {
    A[i] = (T)A_gen[i];
    B[i] = (T)B_gen[i];
    C[i] = 0;
  }

  float time_taken = 0;
  {
      //We'll figure out T data type and call appropriate function : 
      // Int : matmul_1 , Float : matmul_2 and Double : matmul_3
      UnitGPUTime g;
      if (std::is_same<T, int>::value)
        matmul_1((const int*)A, (const int*)B, (int*)C, N, block_dim);
      else if (std::is_same<T, float>::value)
        matmul_2((const float*)A, (const float*)B, (float*)C, N, block_dim);
      else if (std::is_same<T, double>::value)
        matmul_3((const double*)A, (const double*)B, (double*)C, N, block_dim);
      else
      {
        cout << "Wrong matmul called" << endl;
        return 0;
      }
      time_taken = g.getTime();
  }
  cout << C[0] << endl << C[N * N - 1] << endl << time_taken << endl;

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
  cudaCheckError();

  // Free memory
  cudaFree(A);
  cudaFree(B);
  cudaFree(C);  
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

  // Generate Random Values for kernel
  std::random_device entropy_source;
  std::mt19937 generator(entropy_source());
  std::uniform_real_distribution<float> dist(-5, 5);

  // Fill random arrays to be used with A and B for Int, float and double.
  A_gen = (float*)malloc( N * N * sizeof(float) );
  B_gen = (float*)malloc( N * N * sizeof(float) );
  for (size_t i = 0; i < N * N; i++)
  {
    A_gen[i] = dist(generator);
    B_gen[i] = dist(generator);
  }

  // Perform matmul for int float and double
  perf_matmul<int>(N, block_dim );
  perf_matmul<float>(N, block_dim );
  perf_matmul<double>(N, block_dim );

  // free global arrays.
  free(A_gen);
  free(B_gen);
}
