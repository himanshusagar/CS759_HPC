#include "matmul.h"

int getIndex(int size , int i , int j)
{
    return i * size + j;
}

void mmul1(const double* A, const double* B, double* C, const unsigned int n)
{
    for (size_t i = 0; i < n; i++)
    for (size_t j = 0; j < n; j++)
    for (size_t k = 0; k < n; k++)
    {
        C[ getIndex(n , i , j) ] += A[ getIndex(n ,i , k) ] * B[ getIndex(n , k , j) ];
    }
}
void mmul2(const double* A, const double* B, double* C, const unsigned int n)
{
    for (size_t i = 0; i < n; i++)
    for (size_t k = 0; k < n; k++)
    for (size_t j = 0; j < n; j++)
    {
        C[ getIndex(n , i , j) ] += A[ getIndex(n ,i , k) ] * B[ getIndex(n , k , j) ];
    }

    
}

void mmul3(const double* A, const double* B, double* C, const unsigned int n)
{
   
    for (size_t j = 0; j < n; j++)
    for (size_t k = 0; k < n; k++)
    for (size_t i = 0; i < n; i++)
    {
        C[ getIndex(n , i , j) ] += A[ getIndex(n ,i , k) ] * B[ getIndex(n , k , j) ];
    }
}

void mmul4(const std::vector<double>& A, const std::vector<double>& B, double* C, const unsigned int n)
{
    for (size_t i = 0; i < n; i++)
    for (size_t j = 0; j < n; j++)
    for (size_t k = 0; k < n; k++)
    {
        C[ getIndex(n , i , j) ] += A[ getIndex(n ,i , k) ] * B[ getIndex(n , k , j) ];
    }
}