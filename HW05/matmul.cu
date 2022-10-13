#include "matmul.cuh"


template <typename T>
__global__ void matmul_kernel(const T *A, const T *B, T *C, unsigned int N, unsigned int BLOCK_SIZE)
{
    extern __shared__ char smem[];
    T* sData = reinterpret_cast<T *>(smem);

    T* As = sData;
    T* Bs = sData + BLOCK_SIZE * BLOCK_SIZE;

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int aBegin= N * BLOCK_SIZE * by;
    int aEnd= aBegin + N - 1; 
    int aStep= BLOCK_SIZE;

    int bBegin= BLOCK_SIZE * bx;
    int bStep= BLOCK_SIZE * N;

    T Csub = 0;

    for (int a = aBegin, b = bBegin;a <= aEnd;a += aStep, b += bStep) 
    {
        As[ty * BLOCK_SIZE + tx] = A[a + N * ty + tx];
        Bs[ty * BOCK_SIZE + tx] = B[b + N * ty + tx];

        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; ++k)
            Csub += As[ty * BLOCK_SIZE + k] * Bs[k * BLOCK_SIZE + tx];

        __syncthreads();
    }

    int c = N * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + N * ty + tx] = Csub;
}

template <typename T>
__host__ void matmul(const T *A, const T *B, T *C, unsigned int n,  unsigned int block_dim)
{
    // Launch simple kernel on GPU with 2 block and 8 threads.
    float f_n = n;
    dim3 dimBlock( block_dim, block_dim );
    dim3 dimGrid( f_n/dimBlock.x, f_n/dimBlock.y);
    matmul_kernel<T><<< dimGrid, dimBlock >>>(A, B, C, d);
}


__host__ void matmul_1(const int *A, const int *B, int *C, unsigned int n, unsigned int block_dim)
{
    matmul<int>(A , B , C , n, block_dim);
}
__host__ void matmul_2(const float *A, const float *B, float *C, unsigned int n, unsigned int block_dim)
{
    matmul<float>(A , B , C , n, block_dim);
}
__host__ void matmul_3(const double *A, const double *B, double *C, unsigned int n, unsigned int block_dim)
{
    matmul<double>(A , B , C , n, block_dim);
} 


