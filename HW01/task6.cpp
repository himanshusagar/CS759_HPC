#include <iostream>
#include <chrono>

int main() {
    int N;
    std::cin>>N;
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
