#include <iostream>
#include <string>
#include <random>
#include <cmath>

#include "profile.cuh"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/sort.h>
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/uniform_real_distribution.h>


int main(int argc, char *argv[])
{
  if (argc != 2)
  {
    std::cout << "Usage ./task_thrust n " << std::endl;
    return 0;
  }
  size_t N = std::stoi(argv[1]);
  // Generate Random Values for kernel 
  thrust::minstd_rand generator;
  thrust::uniform_real_distribution<float> dist(-1,1);
  thrust::host_vector<float> h_vec(N);

  //Fill host vector with random values
  for (size_t i = 0; i < N; i++)
  {
    h_vec[i] = dist(generator);
  }
  // transfer data to the device
  thrust::device_vector<float> d_vec(N);
  thrust::copy(h_vec.begin(), h_vec.end(), d_vec.begin()); 

  // call kernel and compute time.
  float time_val = 0;
  float sol = 0;
  {
    UnitGPUTime g;
    //Call Reduction Kernel
    sol = thrust::reduce( d_vec.begin(), d_vec.end() );
    time_val = g.getTime();
  }
  //Print out output time.
  std::cout << sol << std::endl << time_val << std::endl;
  
  return 0;
}
