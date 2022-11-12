#include <iostream>
#include <algorithm>    
#include <chrono>
#include <string>
#include <cmath>

#include <omp.h>

#include "profile.h"
#include "cluster.h"

int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        std::cout << "Usage ./task1 n t" << std::endl;
        return 0;
    }
    size_t N = std::stoi(argv[1]);
    size_t T = std::stoi(argv[2]);
    //Set thread count
    omp_set_num_threads(T);

    //Int matrices for multiplication
    float *arr = new float[N];
    float *centers = new float[T];
    float *dists = new float[T];
    
    // Generate random values
    std::default_random_engine e;
    e.seed(std::chrono::system_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<> dis(0, N);

    // Fill values in arr.
    for (size_t i = 0; i < N; i++)
    {
        arr[i] = dis(e);
    }

    //Sort them up.
    std::sort(arr , arr + N);

    float f_N = N;
    float f_T = T;
    
    // Fill centers arrays. 
    for (size_t i = 0; i < T; i++)
    {
        centers[i] =  (2 * i * f_N)  +  (f_N) / (2 * f_T);
        dists[i] = 0;
    }
    
    // Run and compute time taken(ms)
    float time_taken;
    {
        UnitTime u;
        cluster(N , T, arr , centers , dists);
        time_taken = u.getTime();
    }

    float max_tot = -1;
    float max_index = -1;

    //Reduction to find Max Value and Max Index  
    #pragma omp parallel for reduction( max : max_tot)
    for (size_t i = 0; i < T; i++)
    {
        if (dists[i] > max_tot)
        {
            max_tot = dists[i];
            max_index = i;
        }
    }

    //Print out results as per HW.
    std::cout << max_tot << std::endl <<  max_index << std::endl <<  time_taken << std::endl;
    
    delete[] arr;
    delete[] centers;
    delete[] dists;
    return 0;
}



