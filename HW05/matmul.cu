#include "matmul.cuh"
#include "profile.cuh"

template <typename T>
__global__ void matmul_kernel(const T *A, const T *B, T *C, unsigned int N, unsigned int BLOCK_SIZE)
{
    extern __shared__ char smem[];
    T* sData = reinterpret_cast<T *>(smem);

    T* As = sData;
    T* Bs = sData + BLOCK_SIZE * BLOCK_SIZE;

    int tx = threadIdx.x; 
    int ty = threadIdx.y;

    // Global row and col index for bounds checking.
    int row = blockIdx.y * blockDim.y + ty;
    int col = blockIdx.x * blockDim.x + tx;
    T cSol = 0;
    int limit = (N - 1)/BLOCK_SIZE + 1;

    for (int k = 0; k < limit; ++k) 
    {
        // See out of tile for As
        if(row < N && ( (k * BLOCK_SIZE + tx) < N)  )
            As[ty * BLOCK_SIZE + tx] = A[ row * N + k * BLOCK_SIZE + tx];
        else 
            As[ty * BLOCK_SIZE + tx] = 0;

        // See out of tile for As
        if(col < N && ( (k * BLOCK_SIZE + ty) < N ) )
            Bs[ty * BLOCK_SIZE + tx] = B[( k * BLOCK_SIZE + ty ) * N + col]; 
        else
            Bs[ty * BLOCK_SIZE + tx] = 0;
        
        __syncthreads();
        
        // We still need to check here because otherwise As and Bs would not be filled correctly.
        // But we only need to compute cSol for indices in range.
        if(row < N && col < N)
        for (int i = 0; i < BLOCK_SIZE; ++i)
        {  
            cSol += As[ty * BLOCK_SIZE + i] * Bs[i * BLOCK_SIZE + tx];
        }
        __syncthreads();
    }

    // Now fill back results for cSol's computed.
    if(row < N && col < N)
        C[row * N + col] = cSol;
}


template <typename T>
__host__ void matmul(const T *A, const T *B, T *C, unsigned int N,  unsigned int block_dim)
{
    // Launch simple kernel on GPU with 2 block and 8 threads.
    float f_N = N;
    float f_block_dim = block_dim;
    size_t grid_size = ceil(f_N / f_block_dim);
    dim3 dimBlock( block_dim, block_dim );
    dim3 dimGrid( grid_size , grid_size );
    size_t shared_mem_size = 2 * block_dim * block_dim * sizeof(T); // two mini matrices of size block_dim.
    //std::cout << block_dim << "X" << grid_size << " SM: " << shared_mem_size << " " << N << std::endl;
    matmul_kernel<T><<< dimGrid, dimBlock , shared_mem_size >>>(A, B, C, N, block_dim);
    cudaCheckError();
}


__host__ void matmul_1(const int *A, const int *B, int *C, unsigned int n, unsigned int block_dim)
{
    //Int version.
    matmul<int>(A , B , C , n, block_dim);
}
__host__ void matmul_2(const float *A, const float *B, float *C, unsigned int n, unsigned int block_dim)
{
    //Float version.
    matmul<float>(A , B , C , n, block_dim);
}
__host__ void matmul_3(const double *A, const double *B, double *C, unsigned int n, unsigned int block_dim)
{
    //Double version.
    matmul<double>(A , B , C , n, block_dim);
} 


