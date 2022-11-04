#include "convolution.h"

bool inRange(int begin , int value, int end)
{
    //check whether value lies between begin and end
    return begin <= value && value < end;
}
int getIndex(int size , int i , int j)
{
    //return matrix index's in 1-D array
    return i * size + j;
}
void convolve(const float *image, float *output, std::size_t n, const float *mask, std::size_t m)
{
    #pragma omp parallel for collapse(2)
    for (size_t x = 0; x < n; x++)
    {
        for (size_t y = 0; y < n; y++)
        {
            //Perform Convolution for constant x and y - that's why we can use collapse
            float sol = 0;
            for (size_t i = 0; i < m; i++)
            {          
                for (size_t j = 0; j < m; j++)
                {
                    int a = x + i - ((m-1)/2);
                    int b = y + j - ((m-1)/2);
                    //Handle 3 cases for boundry conditions.
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
            //Fill in summed over value
            output[ getIndex(n , x , y) ] = sol;
        }
    }
}