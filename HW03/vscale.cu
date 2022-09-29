
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//vscale kernel to compute element-wise product between two vectors - a and b. Store result in b.
__global__ void vscale(const float *a, float *b, unsigned int n)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(0 <= index && index < n)
        b[index] = a[index] * b[index];
}