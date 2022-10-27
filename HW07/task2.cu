#include <iostream>
#include <string>
#include <random>
#include <cmath>

#include "profile.cuh"
#include "count.cuh"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/sort.h>


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
  std::random_device entropy_source;
  std::mt19937 generator(entropy_source());
  std::uniform_int_distribution<> dist(0, 500);
  thrust::host_vector<int> h_vec(N);

  for (size_t i = 0; i < N; i++)
  {
    h_vec[i] = dist(generator);
  }
  // transfer data to the device
  thrust::device_vector<int> d_vec(N) , values, counts;
  thrust::copy(h_vec.begin(), h_vec.end(), d_vec.begin()); 


  // call kernel and compute time.
  float time_val = 0;
  //float sol = 0;
  {
    UnitGPUTime g;
    //sol = 
    count(d_vec , values , counts);
    time_val = g.getTime();
  }
  
  //std::cout << sol << std::endl << time_val << std::endl;
  std::cout << std::log2(N) << "," << time_val << std::endl;

  return 0;
}
