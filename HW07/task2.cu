#include <iostream>
#include <string>
#include <cmath>

#include "profile.cuh"
#include "count.cuh"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/sort.h>
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/uniform_int_distribution.h>


void printX( thrust::device_vector<int>& val)
{
  std::cout << "Array: " << std::endl;
  thrust::copy(val.begin(), val.end(), std::ostream_iterator<int>(std::cout, " "));
  std::cout << std::endl;
}

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
  thrust::uniform_int_distribution<int> dist(0,500);
  thrust::host_vector<int> h_vec(N);

  //Fill host vector with values
  for (size_t i = 0; i < N; i++)
  {
    h_vec[i] = dist(generator);
  }
  // transfer data to the device
  thrust::device_vector<int> d_vec(N) , values, counts;
  thrust::copy(h_vec.begin(), h_vec.end(), d_vec.begin()); 

  // call kernel and compute time.
  float time_val = 0;
  {
    UnitGPUTime g;
    //Call Count Kernel
    count(d_vec , values , counts);
    time_val = g.getTime();
  }
  int last = values.size(); 
  last--;
  //Print out last elements.
  std::cout << values[last] << std::endl;
  std::cout << counts[last] << std::endl;
  std::cout << time_val << std::endl;
  return 0;
}
