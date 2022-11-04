#include <iostream>
#include <omp.h>
#include <chrono>
#include <string>
#include <cmath>

#include "profile.h"
#include "convolution.h"

using namespace std;

int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        cout << "Usage ./task2 N T" << endl;
        return 0;
    }
    size_t N = atoi(argv[1]);
    size_t M = 3;
    size_t T = atoi(argv[2]);
    //Set thread count
    omp_set_num_threads(T);

    //Allocate memory
    float *image = new float[N * N];
    float *output = new float[N * N];
    float *mask = new float[M * M];

    //Generate random values
    std::default_random_engine e;
    e.seed(std::chrono::system_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<> image_dis(-10, 10);
    std::uniform_real_distribution<> mask_dis(-1, 1);

    //Fill random values and init output
    for (size_t i = 0; i < N * N; i++)
    {
        image[i] = image_dis(e);
        output[i] = 0;
    }
    for (size_t i = 0; i < M * M; i++)
        mask[i] = mask_dis(e);

    float time_taken;
    {
        UnitTime u;
        convolve(image, output, N, mask, M);
        time_taken = u.getTime();
    }
    //Print out results as per HW.
    cout << output[0] << endl << output[(N * N) - 1] << endl << time_taken <<  endl;

    delete[] output;
    delete[] image;
    delete[] mask;
    return 0;
}
