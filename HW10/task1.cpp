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
    std::cout << *dest << std::endl <<  time_taken << std::endl;
}

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        std::cout << "Usage ./task1 n" << std::endl;
        return 0;
    }
    size_t N = std::stoi(argv[1]);

    // Generate RANDS array so as to avoid overflow.
    std::vector<int> RANDS(N , 1);
    // Set last element to 2
    RANDS[N - 1] = 2;

    // Generate random indices
    std::default_random_engine gen;
    gen.seed(std::chrono::system_clock::now().time_since_epoch().count());
    std::uniform_int_distribution<> dis(0, N - 1);

    vec arr(N);
    arr.data = new data_t[N]; 
    data_t *dest = new data_t;

    // Fill values in arr.
    for (size_t i = 0; i < N; i++)
    {
        //Pick random value from RANDS array
        arr.data[i] = RANDS[ dis(gen) ];
    }

    // Function pointer table
    void (*optimize_array[6])(vec *, data_t *) { NULL , optimize1 , optimize2 , optimize3 , optimize4 , optimize5 };

    // Iterate over function pointers to pick optimizeX function 
    // and run over 10 times to compute average.
    for(int i = 1 ; i < 6 ; i++)
    {
        float time_taken = 0;
        for (int _ = 0; _ < 10; _++)
        {
            UnitTime u;
            (*optimize_array[ i ]) ( &arr , dest);
            time_taken += u.getTime();
        }
        // Take average
        time_taken = time_taken / 10.0;
        print_it(dest , time_taken);
    }

    delete dest;
    delete[] arr.data;
    return 0;
}



