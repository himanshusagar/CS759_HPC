#include <iostream>
#include <algorithm>
#include <cmath>
#include <vector>

#include "msort.h"



void merge(int *arr, int l,
		int m, int r)
{
	int i, j, k;
	int n1 = m - l + 1;
	int n2 = r - m;

	// Create temp arrays
	int L[n1], R[n2];

	// Copy data to temp arrays
	// L[] and R[]
	for (i = 0; i < n1; i++)
		L[i] = arr[l + i];
	for (j = 0; j < n2; j++)
		R[j] = arr[m + 1 + j];

	// Merge the temp arrays back
	// into arr[l..r]
	// Initial index of first subarray
	i = 0;

	// Initial index of second subarray
	j = 0;

	// Initial index of merged subarray
	k = l;
	while (i < n1 && j < n2)
	{
		if (L[i] <= R[j])
		{
			arr[k] = L[i];
			i++;
		}
		else
		{
			arr[k] = R[j];
			j++;
		}
		k++;
	}

	// Copy the remaining elements
	// of L[], if there are any
	while (i < n1) {
		arr[k] = L[i];
		i++;
		k++;
	}

	// Copy the remaining elements of
	// R[], if there are any
	while (j < n2)
	{
		arr[k] = R[j];
		j++;
		k++;
	}
}

// l is for left index and r is
// right index of the sub-array
// of arr to be sorted
void mergeSort(int *arr,
			int l, int r)
{
	if (l < r)
	{
		// Same as (l+r)/2, but avoids
		// overflow for large l and h
		int m = l + (r - l) / 2;

		// Sort first and second halves
		mergeSort(arr, l, m);
		mergeSort(arr, m + 1, r);

		merge(arr, l, m, r);
	}
}


void unit_sort(int* arr, int begin , int end , int N)
{
    if( (begin < end) && (end < N) )
    {
        end = std::min( N-1 , end);
        int size = end - begin + 1;
        std::sort( arr + begin, arr + size);
    }
}

void unit_merge(int* arr, int begin1 , int end1 , int begin2 , int end2 )
{
    std::vector<int> buffer;
    int init_begin = begin1;
    while(begin1 < end1 && begin2 < end2)
    {
        if(arr[begin1] < arr[begin2])
        {
            buffer.push_back( arr[begin1++] );
        }
        else
        {
            buffer.push_back( arr[begin2++] );
        }

    }
    while(begin1 < end1)
    {
        buffer.push_back( arr[begin1++] );
    }
    while(begin2 < end2)
    {
        buffer.push_back( arr[begin2++] );
    }
    std::copy(buffer.begin(), buffer.end(),
              arr + init_begin);
    
}
int get_half_ceil(int a)
{
    int f_a = a;
    return ceil( f_a/2.0f );
}
void merge_results(int* arr, float N , float T)
{
    //Time to merge
    int level = std::ceil(  std::log2(T) );  // T elements to be merged;
    while (level--)
    {
        int portion_size = std::ceil( N/T );
        #pragma omp parallel for
        for(int i = 0; i < get_half_ceil(T) ; i++)
        {
            int begin = i * portion_size;
            int end  = begin + portion_size - 1;
            unit_merge(arr , begin , end , end + 1 , end + portion_size - 1 );

        }
        T = get_half_ceil(T); 
    }
    
}
void msort(int* arr, const std::size_t n, const std::size_t threshold)
{
    if(n < threshold)
    {
        unit_sort(arr, 0 , n-1 , n);
        return;
    }

    float N = n;
    int T = omp_get_num_threads();
    // Individual sort done.
    {
        int portion_size = std::ceil( N/T );
        #pragma omp parallel for
        for (int i = 0; i < T ; i++)
        {
            int begin = i * portion_size;
            int end = begin + portion_size - 1;
            unit_sort(arr, begin , end , n);
        }
    }

    merge_results(arr, N , T);
}