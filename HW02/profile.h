#ifndef PROFILE_H
#define PROFILE_H

#include <iostream>
#include <chrono>
#include <vector>
#include <string>
#include <numeric>
#include <functional>
#include <random>

struct UnitTime
{
    std::chrono::high_resolution_clock::time_point begin;
    UnitTime() : begin(std::chrono::high_resolution_clock::now()) { }
    ~UnitTime()
    {
        auto d = std::chrono::high_resolution_clock::now() - begin;
        uint64_t countValue = std::chrono::duration_cast<std::chrono::microseconds>(d).count();
        std::cout << countValue << ",";
    }
};
#endif //PROFILE_H
