#include <iostream>
#include <algorithm>    
#include <chrono>
#include <string>
#include <cmath>
#include <random>

#include "optimize.h"
#include "profile.h"

void print_it(data_t *dest, float time_taken )
{
    std::cout << time_taken << std::endl;
}

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        std::cout << "Usage ./task1 N" << std::endl;
        return 0;
    }
    size_t N = std::stoi(argv[1]);
    
    // Generate random values
    std::default_random_engine gen;
    gen.seed(std::chrono::system_clock::now().time_since_epoch().count());
    std::uniform_int_distribution<> dis(-10, 10);

    vec arr(N);
    arr.data = new data_t[N]; 
    data_t *dest = new data_t;

    // Fill values in arr.
    for (size_t i = 0; i < N; i++)
    {
        arr.data[i] = dis(gen);
    }

    // Time to monte carlo
    {
        float time_taken;
        {
            UnitTime u;
            optimize1( &arr , dest);
            time_taken = u.getTime();
        }
        print_it(dest , time_taken);
       
    }
    {
        float time_taken;
        {
            UnitTime u;
            optimize2( &arr , dest);
            time_taken = u.getTime();
        }
        print_it(dest , time_taken);
    }
    {
        float time_taken;
        {
            UnitTime u;
            optimize3( &arr , dest);
            time_taken = u.getTime();
        }
        print_it(dest , time_taken);
    }
    {
        float time_taken;
        {
            UnitTime u;
            optimize4( &arr , dest);
            time_taken = u.getTime();
        }
        print_it(dest , time_taken);
    }
    {
        float time_taken;
        {
            UnitTime u;
            optimize5( &arr , dest);
            time_taken = u.getTime();
        }
        print_it(dest , time_taken);
    }
    
    delete dest;
    delete[] arr.data;
    return 0;
}



