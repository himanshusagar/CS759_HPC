#include <iostream>

#include "profile.cuh"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/execution_policy.h>

void count(const thrust::device_vector<int>& d_in,
    thrust::device_vector<int>& values,
    thrust::device_vector<int>& counts)
{    
    
    thrust::device_vector<int> d_sorted_in(d_in);
    //Sort before reduce
    thrust::sort(d_sorted_in.begin(), d_sorted_in.end());
    //Unique number of elements
    size_t U_N = thrust::inner_product(d_sorted_in.begin(), d_sorted_in.end() - 1, d_sorted_in.begin() + 1,
                         0, thrust::plus<int>(), thrust::not_equal_to<int>()) + 1;
    // Init vectors.
    thrust::device_vector<int> op_key(U_N);
    thrust::device_vector<int> op_value(U_N);
    // Perform reduction.
    thrust::reduce_by_key(d_sorted_in.begin(), d_sorted_in.end(), thrust::constant_iterator<int>(1)
                 , op_key.begin(), op_value.begin());  
    // Fill Both Vectors.
    values = thrust::device_vector<int>(op_key);
    counts = thrust::device_vector<int>(op_value);
}

