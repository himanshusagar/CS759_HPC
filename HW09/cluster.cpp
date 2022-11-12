#include "cluster.h"
#include <cmath>
#include <iostream>

void cluster(const size_t n, const size_t t, const float *arr,
             const float *centers, float *dists) {
#pragma omp parallel num_threads(t)
  {
    unsigned int tid = omp_get_thread_num();
    //Perform Reduction over dists[tid] to remove false sharing problem.
#pragma omp for reduction( + : dists[tid])
    for (size_t i = 0; i < n; i++) 
    {
      dists[tid] += std::fabs(arr[i] - centers[tid]);
    }
  }
}
