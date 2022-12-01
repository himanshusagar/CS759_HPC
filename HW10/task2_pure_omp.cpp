#include <iostream>
#include <algorithm>    
#include <chrono>
#include <string>
#include <cmath>
#include <cstring>

#include "profile.h"
#include "reduce.h"


int main(int argc, char *argv[])
{
    size_t arr_size = std::stoi(argv[1]);
    size_t T = std::stoi(argv[2]);
    
    //Set thread count
    omp_set_num_threads(T);
    float *array = new float[arr_size];
    float global_res = 0;

    // Generate random values
    std::default_random_engine e;
    e.seed(std::chrono::system_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<> dis(-1, 1);

    //Fill buffers with random values
    for (size_t i = 0; i < arr_size; i++)
    {
        array[i] = dis(e);
    }

    float time_taken;
    {
        UnitTime u;
        global_res = reduce(array , 0 , arr_size);
        time_taken = u.getTime();
    }

    {
        //std::cout << time_taken <<  std::endl;
        std::cout << std::log2(arr_size) << ", " << T << ", " <<  time_taken << std::endl;
    }

    return 0;
}
