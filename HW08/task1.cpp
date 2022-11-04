#include <iostream>
#include <omp.h>
#include <chrono>
#include <string>
#include <cmath>

#include "profile.h"
#include "matmul.h"

int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        std::cout << "Usage ./task1 n t" << std::endl;
        return 0;
    }
    size_t N = std::stoi(argv[1]);
    size_t T = std::stoi(argv[2]);
    omp_set_num_threads(T);

    float *A = new float[N * N];
    float *B = new float[N * N];
    float *C = new float[N * N];

    std::default_random_engine e;
    e.seed(std::chrono::system_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<> dis(-10, 10);

    // std::cout << T << ", ";
    for (size_t i = 0; i < N * N; i++)
    {
        A[i] = dis(e);
        B[i] = dis(e);
        C[i] = 0;
    }

    float time_taken;
    {
        UnitTime u;
        mmul(A, B, C, N);
        time_taken = u.getTime();
    }
    std::cout << C[0] << std::endl <<  C[N * N - 1] << std::endl <<  time_taken << std::endl;

    delete[] A;
    delete[] B;
    delete[] C;
    return 0;
}



