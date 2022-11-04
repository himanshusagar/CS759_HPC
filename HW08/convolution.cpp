#include "convolution.h"

bool inRange(int begin , int value, int end)
{
    return begin <= value && value < end;
}
int getIndex(int size , int i , int j)
{
    return i * size + j;
}
void convolve(const float *image, float *output, std::size_t n, const float *mask, std::size_t m)
{
    #pragma omp parallel for collapse(2)
    for (size_t x = 0; x < n; x++)
    {
        for (size_t y = 0; y < n; y++)
        {
            //Perform Convolution
            float sol = 0;
            for (size_t i = 0; i < m; i++)
            {          
                for (size_t j = 0; j < m; j++)
                {
                    int a = x + i - ((m-1)/2);
                    int b = y + j - ((m-1)/2);

                    float value = 0;
                    if( !inRange(0 , a , n) && !inRange(0 , b , n))
                        value = 0;
                    else if( !inRange(0 , a , n) || !inRange(0 , b , n))
                        value = 1;
                    else
                        value = image[ getIndex(n , a , b) ];
                    
                    sol += mask[ getIndex(m , i , j) ] * value;
                }
            }
            output[ getIndex(n , x , y) ] = sol;
        }
    }
}