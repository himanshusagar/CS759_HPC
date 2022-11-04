#include <iostream>
#include <algorithm>
#include <cmath>
#include <vector>

#include "msort.h"

using std::vector;

void insertionSort(vector<int>& vec, int begin, int end) 
{
    //Simple insertion sort implementation for vec size less than threshold
    for(int j = begin + 1; j <= end; j++)
    {
      int key = vec[j];
      int i = j-1;

      while(i >= begin && vec[i] > key)
      {
         vec[i+1] = vec[i];
         i--;
      }
      vec[i+1] = key;
    }
}

void merge_results(vector<int>& vec, int begin1 , int end1 , int begin2 , int end2 )
{
    //Use Empty buffer to put back results in sorted order.
    std::vector<int> buffer;
    int init_begin = begin1;
    while(begin1 <= end1 && begin2 <= end2)
    {
        if(vec[begin1] < vec[begin2])
        {
            buffer.push_back( vec[begin1++] );
        }
        else
        {
            buffer.push_back( vec[begin2++] );
        }

    }
    //Pushback values left from left side
    while(begin1 <= end1)
    {
        buffer.push_back( vec[begin1++] );
    }
    //Pushback values left from right side
    while(begin2 <= end2)
    {
        buffer.push_back( vec[begin2++] );
    }
    //Fill vec with results.
    std::copy(buffer.begin(), buffer.end(),
              vec.begin() + init_begin);
    
}

void mergeSort(vector<int>& vec, int begin, int end, int threshold)
{
    // Check if out of range.
    if (begin >= end)
        return;
    // Check if insertion sort needs to be used.
    if( (end - begin) <= threshold) 
    {
        insertionSort(vec , begin , end);
        return;
    }   
    //Find mid and perform sort.
    int mid = (begin + end)/2; 
    #pragma omp taskgroup
    {
        #pragma omp task shared(vec) untied
        mergeSort(vec, begin, mid, threshold);
        #pragma omp task shared(vec) untied
        mergeSort(vec, mid + 1, end, threshold);
        #pragma omp taskyield
    }
    merge_results(vec,  begin,  mid , mid + 1, end );
}

void msort(int* arr, const std::size_t n, const std::size_t threshold)
{
    #pragma omp parallel
    #pragma omp single
    {
        //Make vector out of array, perform sort and copy back results.
        vector <int> vec(arr , arr + n);
        mergeSort(vec, 0 , n - 1 , threshold);
        std::copy(vec.begin(), vec.end(), arr);
    }
}