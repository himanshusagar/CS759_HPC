#include <iostream>
#include <string>
#include <random>
#include <cmath>
#include <omp.h>

int main(int argc, char *argv[])
{
    //Set thread count
    omp_set_num_threads(4);
    std::printf("Number of threads: %d\n", omp_get_max_threads());

#pragma omp parallel
    {
    //Get thread count
    int myId = omp_get_thread_num();
    //Print thread id.
    std::printf("I am thread No: %d\n", myId );
    }
//Parallel For for distributing tasks among 4 threads.
#pragma omp parallel for
    for(int i = 1 ; i <= 8; i++)
    {
        //Generate factorial
        int sol = 1;
        for(int j = 1 ; j <= i; j++)
        {
            sol *= j;
        }
        std::printf("%d!=%d\n", i, sol);
        fflush(stdout);
    }
    return 0;
}