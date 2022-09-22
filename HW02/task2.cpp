#include <iostream>
#include "profile.h"
#include "convolution.h"

using namespace std;

int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        cout << "Usage ./task2 N M" << endl;
        return 0;
    }
    size_t N = atoi(argv[1]);
    size_t M = atoi(argv[2]);

    float *image = new float[N * N];
    float *output = new float[N * N];
    float *mask = new float[M * M];

    std::default_random_engine e;
    e.seed(std::chrono::system_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<> image_dis(-10, 10);
    std::uniform_real_distribution<> mask_dis(-1, 1);

    // float img[] = {1,3,4,8,
    //                 6,5,2,4,
    //                 3,4,6,8,
    //                 1,4,5,2};
    // float msk[] = {
    //                 0, 0, 1,
    //                 0, 1, 0,
    //                 1, 0, 0};
    for (size_t i = 0; i < N * N; i++)
        image[i] = image_dis(e);
    for (size_t i = 0; i < M * M; i++)
        mask[i] = mask_dis(e);
    {
        UnitTime u;
        convolve(image, output, N, mask, M);
    }
    cout << output[0] << endl << output[(N * N) - 1] << endl;

    delete[] output;
    delete[] image;
    delete[] mask;
    return 0;
}
