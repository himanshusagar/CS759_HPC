#include <iostream>
#include "profile.h"
#include "scan.h"

using namespace std;

int main(int argc, char *argv[])  
{
    if(argc != 2)
    {
        cout << "Usage ./task6 <Some Positive Integer>" << endl;
        return 0;
    }
    size_t N = atoi(argv[1]);
    float *arr = new float[N];
    float *output = new float[N];

    std::default_random_engine e;
    e.seed(std::chrono::system_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<> dis(-1, 1);
    cout << log2(N) << ",";
    for (size_t i = 0; i < N; i++)
    {
        arr[i] = dis(e);
    }
    {
        UnitTime u;
        scan(arr, output, N);
    }
    cout << output[0] << "," << output[N-1] << endl;
    delete[] arr;
    delete[] output;

	return 0;
}
