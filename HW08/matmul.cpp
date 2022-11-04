#include "matmul.h"

//Macro to calculate position in 1-D array.
#define GET_INDEX(size, i, j) i * size + j

void mmul(const float* A, const float* B, float* C, const std::size_t n)
{
    #pragma omp parallel for
    for (size_t i = 0; i < n; i++)
    {
        for (size_t k = 0; k < n; k++)
        {    
            for (size_t j = 0; j < n; j++)
            {
                //For given i and j -> sum over A and B to compute rs
                C[ GET_INDEX(n , i , j) ] += A[ GET_INDEX(n ,i , k) ] * B[ GET_INDEX(n , k , j) ];
            }
        }
    }
}
