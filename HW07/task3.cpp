#include <iostream>
#include <string>
#include <random>
#include <cmath>
#include <omp.h>

int main(int argc, char *argv[])
{

    omp_set_num_threads(4);
    std::printf("Number of threads: %d\n", omp_get_max_threads());

#pragma omp parallel
{
    int myId = omp_get_thread_num();
    std::printf("I am thread No: %d\n", myId );
    fflush(stdout); 
    int sol = 1;
    for(int i = 1 ; i <= 8; i++)
    {
        sol *= i; 
        std::printf("%d!=%d\n", i, sol);
        fflush(stdout);
    }
}

    return 0;
}