#include <iostream>
#include <algorithm>    
#include <chrono>
#include <string>
#include <cmath>

#include <omp.h>

#include "profile.h"
#include "montecarlo.h"

int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        std::cout << "Usage ./task2 n t" << std::endl;
        return 0;
    }
    float R = 1.0;
    size_t N = std::stoi(argv[1]);
    size_t T = std::stoi(argv[2]);
    //Set thread count
    omp_set_num_threads(T);

    //Int matrices for multiplication
    float *x = new float[N];
    float *y = new float[N];

    // Generate random values
    std::default_random_engine e;
    e.seed(std::chrono::system_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<> dis(-R, R);

    // Fill values in arr.
    for (size_t i = 0; i < N; i++)
    {
        x[i] = dis(e);
        y[i] = dis(e);
    }

    float in_points = 0;
    float tot_points = N;
    
    // Time to monte carlo
    float time_taken;
    {
        UnitTime u;
        in_points = montecarlo(N , x , y , R);
        time_taken = u.getTime();
    }

    float PI_VAL = (4 * in_points) / tot_points;

    //Print out results as per HW.
    std::cout << PI_VAL << std::endl << time_taken << std::endl;
    
    delete[] x;
    delete[] y;
    return 0;
}



