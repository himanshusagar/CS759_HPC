#include "mmul.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "profile.cuh"

#include <iostream>
#include <string>
#include <random>

using std::cout;
using std::endl;

void printX(float *val, int N)
{
    std::cout << "Mat : " << std::endl;
    for(int i = 0; i < N ; i++)
    {
      for(int j = 0; j < N ; j++)
      {
        std::cout << val[i * N + j] << " ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
}



int main(int argc, char *argv[])
{
  if (argc != 3)
  {
    cout << "Usage ./task1 n n_tests" << endl;
    return 0;
  }
  size_t N = std::stoi(argv[1]);
  size_t N_tests = std::stoi(argv[2]);

  // Generate Random Values for kernel
  std::random_device entropy_source;
  std::mt19937 generator(entropy_source());
  std::uniform_real_distribution<float> dist(-5, 5);

  float *A, *B, *C, *C_temp;
  size_t size = N * N * sizeof(float);
  // Allocate Unified Memory -- accessible from CPU or GPU
  cudaMallocManaged(&A, size);
  cudaCheckError();
  cudaMallocManaged(&B, size);
  cudaCheckError();
  cudaMallocManaged(&C, size);
  cudaCheckError();
  cudaMallocManaged(&C_temp, size);
  cudaCheckError();
    

  cublasHandle_t handle;
  cublasCreate(&handle);

  // initialize A,B and C matrices on the host and gpu
  for (size_t i = 0; i < N * N; i++)
  {
    A[i] = i;
    C_temp[i] = i;
  }

  for (size_t i = 0; i < N; i++)
  {
    B[i + i * N] = 1;
  }

  std::vector<float> values;
  for(size_t _ = 0 ; _ < N_tests ; _++)
  {
    cudaMemcpy(C , C_temp , size , cudaMemcpyDeviceToDevice);
    {
      UnitGPUTime g;
      mmul(handle, A, B, C, N);
      values.push_back(g.getTime());
    }
  }

  double tot = std::accumulate(values.begin() , values.end() , 0.f);
  double count = N_tests;
  //std::cout << tot << " " <<  count << endl;
  std::cout << tot / count << endl;

  // printX(C , N);
  // free unified arrays.
  cublasDestroy(handle);
  cudaFree(A);
  cudaFree(B);
  cudaFree(C);
  cudaFree(C_temp);
  return 0;
}
