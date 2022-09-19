#include <iostream>
#include "profile.h"
#include "matmul.h"

using namespace std;

void init_mat(double* C, int N)
{
    double value = 0;
    for (size_t i = 0; i < N*N; i++)
    { 
        value += C[i];
        C[i] = 0;
    }
    cout << "C = " <<  value << endl;
}

int main(int argc, char *argv[])  
{
    size_t N = 1024;
    
    double *A = new double[N*N];
    double *B = new double[N*N];
    double *C = new double[N*N];
    
    std::default_random_engine e;
    e.seed(std::chrono::system_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<> dis(-10, 10);

    cout << log2(N) << endl;
    for (size_t i = 0; i < N*N; i++)
        A[i] = dis(e);
    for (size_t i = 0; i < N*N; i++)
        B[i] = dis(e);

    {
        init_mat(C , N);
        UnitTime u;
        mmul1(A, B, C, N);
        cout << C[N*N -1] << endl;
    }
    {
        init_mat(C , N);
        UnitTime u;
        mmul2(A, B, C, N);
        cout << C[N*N -1] << endl;
    }
    {
        init_mat(C , N);
        UnitTime u;
        mmul3(A, B, C, N);
        cout << C[N*N -1] << endl;
    }

    delete[] A;
    delete[] B;
    delete[] C;
	return 0;
}
