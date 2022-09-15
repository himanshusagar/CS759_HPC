#include <iostream>
#include <chrono>

int main(int argc, char *argv[])  
{
    if(argc != 2)
    {
        std::cout << "Usage ./task6 <Some Positive Integer>" << std::endl;
        return 0;
    }
    int N = std::atoi(argv[1]);
    if(N < 0)
    {
        std::cout << "Usage ./task6 <Some Positive Integer>" << std::endl;
        return 0;
    }
    for(int i = 0 ; i <= N ; i++)
    {
        printf("%d ", i);
    }
    printf("\n");
    for(int i = N ; i >= 0 ; i--)
    {
        std::cout << i << " ";
    }
    std::cout << std::endl;
    return 0;
}
