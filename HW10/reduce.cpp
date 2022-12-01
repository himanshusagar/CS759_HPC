#include "reduce.h"
#include "optimize.h"

float reduce(const float* arr, const size_t l, const size_t r)
{
    // Set sol as answer variable
    // Run reduction using reduction of openmp
    float sol = 0;
    #pragma omp parallel for simd reduction( + : sol ) 
    for(size_t i = l ; i < r ; i++)
    {
       sol += arr[i];
    }
    return sol;
}