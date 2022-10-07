#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "stencil.cuh"
#include "profile.cuh"

#include <iostream>
#include <string> 
#include <random>

using std::cout;
using std::endl;

void cpu_stencil(const float* image ,const float* mask , float* out_cpu, int N, int R)
{
    float img_val;
    int pos_R = R;
    int neg_R = pos_R * -1;
    for(int i = 0 ; i < N ; i++)
    {
        out_cpu[i] = 0;
        for(int j = neg_R ; j <= pos_R ; j++ )
        {
            int i_j = i + j;
            if( (0 <= i_j) && (i_j < N) )
                img_val = image[i_j];
            else
                img_val = 1.0;
            out_cpu[i] += img_val * mask[j + R];
            
        }
    }
}
int main(int argc, char *argv[])
{
    if (argc != 4)
    {
        cout << "Usage ./task1 n R threads per block" << endl;
        return 0;
    }
    size_t N = std::stoi(argv[1]);
    size_t R = std::stoi(argv[2]);
    size_t threads_per_block = std::stoi(argv[3]);

    float *image, *mask, *output;     
    float *d_image, *d_mask, *d_output;
    size_t image_size = N * sizeof(float);
    size_t mask_size = (2 * R + 1) * sizeof(float);
    

    cudaError_t cudaStatus;
    // Generate Random Values for kernel
    std::random_device entropy_source;
    std::mt19937 generator(entropy_source()); 
    std::uniform_real_distribution<float> dist(-1.0,1.0);
    
    // Allocate space for device and host array a
    cudaMalloc((void **)&d_image, image_size);
    cudaMalloc((void **)&d_mask, mask_size);
    cudaMalloc((void **)&d_output, image_size);
    
    image = (float *)malloc(image_size);
    mask = (float *)malloc(mask_size);
    output = (float *)malloc(image_size);
    //float* out_cpu = (float *)malloc(image_size);
    
    // Fill a, b, c array on host
    for(size_t i = 0; i < N ; i++)
    {
        image[i] = dist(generator);
        output[i] = 0;
    }
    for(size_t i = 0; i < (2 * R + 1) ; i++)
    {
        mask[i] = dist(generator);
    }

    //Copy data from host to device
    cudaMemcpy(d_image, image, image_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, mask, mask_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, output, image_size, cudaMemcpyHostToDevice);
    float time_taken = 0;
    {
        UnitGPUTime g;
        stencil(d_image , d_mask , d_output, N , R , threads_per_block);
        time_taken = g.getTime();
    }
    // Copy result back to host
    cudaStatus = cudaMemcpy(output, d_output, image_size, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy returned error code %d after copying from kernel!\n", cudaStatus);
        return 0;
    }

    cout << output[N - 1] << "," << std::log2(N) << "," << time_taken << endl;

    // cpu_stencil(image , mask ,out_cpu, N,  R);
    // for(size_t i = 0 ; i < N ; i++)
    // {
    //     if( abs( out_cpu[i] - output[i] ) > 1e-5 )
    //     {
    //         cout << "Diff at " << i << " C:" <<  out_cpu[i] << " G:" <<  output[i] << endl;
    //         //break;
    //     }
    // }

    // Cleanup
    free(image);
    free(mask);
    free(output);
    cudaFree(d_image);
    cudaFree(d_mask);
    cudaFree(d_output);
    return 0;
}