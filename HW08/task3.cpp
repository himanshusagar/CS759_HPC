#include <iostream>
#include "profile.h"
#include "msort.h"

using namespace std;

int main(int argc, char *argv[])
{
    if (argc != 4)
    {
        cout << "Usage ./task3 n t ts" << endl;
        return 0;
    }
    size_t N = stoi(argv[1]);
    size_t T = stoi(argv[2]);
    size_t threshold = stoi(argv[3]);

    omp_set_num_threads(T);
    int *arr = new int[N];

    std::default_random_engine e;
    e.seed(std::chrono::system_clock::now().time_since_epoch().count());
    std::uniform_int_distribution<> dis(-1000, 1000);

    for (size_t i = 0; i < N; i++)
        arr[i] = dis(e);
    
    float time_taken;
    {
        UnitTime u;
        msort(arr, N, threshold);
        time_taken = u.getTime();

    }
    
    //cout << N << ", " << T << ", " << std::log2(threshold) << ", " << time_taken << endl;
    cout << arr[0] << endl << arr[N - 1] << endl << time_taken << endl;

    delete[] arr;
    return 0;
}