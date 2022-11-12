#include "montecarlo.h"

int montecarlo(const size_t n, const float *x, const float *y, const float radius)
{
    int in_points = 0;
    #pragma omp parallel for simd reduction( + : in_points ) 
    for(size_t i = 0; i < n ; i++)
    {
        //Compute distance from centre of circle and see if it's less than radius square.
        float dist = x[i] * x[i] + y[i] * y[i];
        if (dist <= radius * radius)
            in_points++;
    }
    return in_points;
}