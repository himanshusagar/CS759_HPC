#include <iostream>
#include <string>
#include <random>
#include "profile.cuh"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

using std::cout;
using std::endl;

int main(int argc, char *argv[])
{
  if (argc != 2)
  {
    cout << "Usage ./task_thrust n " << endl;
    return 0;
  }
  size_t N = std::stoi(argv[1]);
  // Generate Random Values for kernel 
  std::random_device entropy_source;
  std::mt19937 generator(entropy_source());
  std::uniform_real_distribution<float> dist(-1, 1);
  thrust::host_vector<float> h_vec(N);

  for (size_t i = 0; i < N; i++)
  {
    h_vec[i] = i;
  }
  // transfer data to the device
  thrust::device_vector<float> d_vec(N);
  thrust::copy(h_vec.begin(), h_vec.end(), d_vec.begin()); 

  // call kernel and compute time.
  float time_val = 0;
  float sol = 0;
  {
    UnitGPUTime g;
    sol = thrust::reduce( d_vec.begin(), d_vec.end(), 0 , thrust::maximum<float>());
    time_val = g.getTime();
  }
  std::cout << sol << std::endl << time_val << std::endl;

  return 0;
}
