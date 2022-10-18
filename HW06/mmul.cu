#include <iostream>
#include <chrono>
#include "mmul.h"
#include "profile.cuh"

static const char * __cublasGetErrorEnum(cublasStatus_t error)
{
    switch (error)
    {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";

        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";

        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";

        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";

        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";

        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";

        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";

        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";

        case CUBLAS_STATUS_NOT_SUPPORTED:
            return "CUBLAS_STATUS_NOT_SUPPORTED";

        case CUBLAS_STATUS_LICENSE_ERROR:
            return "CUBLAS_STATUS_LICENSE_ERROR";
    }

    return "<unknown>";
}

void mmul(cublasHandle_t handle, const float* A, const float* B, float* C, int N)
{   
    const float alpha = 1.0f;
    const float beta  = 1.0f;

    cublasStatus_t status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, 
        &alpha, A, N, B, N, &beta, C, N);
    if(status != CUBLAS_STATUS_SUCCESS)
    {
        printf("cublasSgemm error %s" , __cublasGetErrorEnum(status));
    }
    cudaDeviceSynchronize();
    cudaCheckError();
}