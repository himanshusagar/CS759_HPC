#include <iostream>
#include <algorithm>    
#include <chrono>
#include <string>
#include <cmath>
#include <cstring>

#include "mpi.h"

#include "profile.h"
#include "reduce.h"


int main(int argc, char *argv[])
{
    int my_rank;       /* rank of process      */
    int p;             /* number of processes  */
    int ret_code = 0;  /* return code for MPI calls */
    float global_res  = 0;

    MPI_Init(&argc, &argv); // Start up MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank); // Find out process rank
    MPI_Comm_size(MPI_COMM_WORLD, &p); // Find out number of processes

    size_t arr_size = std::stoi(argv[1]);
    size_t T = std::stoi(argv[2]);

    //Set thread count
    omp_set_num_threads(T);
    float *array = new float[arr_size];
    
    // Generate random values
    std::default_random_engine e;
    e.seed(std::chrono::system_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<> dis(-1, 1);

    //Fill buffers with random values
    for (size_t i = 0; i < arr_size; i++)
    {
        array[i] = (my_rank + 1);
    }

    ret_code = MPI_Barrier (MPI_COMM_WORLD); 
    if (ret_code != MPI_SUCCESS)
    {
        fprintf(stderr, "MPI_Barrier failed\n");
        exit(0);
    }

    float time_taken;
    {
        UnitTime u;
        float res = reduce(array , 0 , arr_size);
        ret_code = MPI_Reduce(&res, &global_res, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
        if (ret_code != MPI_SUCCESS)
        {
            fprintf(stderr, "MPI_Reduce failed\n");
            exit(0);
        }
        time_taken = u.getTime();
    }

    if(my_rank == 0) 
    {
        //std::cout << time_taken <<  std::endl;
        std::cout << std::log2(arr_size) << ", " << T << ", " <<  time_taken << std::endl;
    }

    ret_code = MPI_Finalize();
    if (ret_code != MPI_SUCCESS)
    {
        fprintf (stderr, "MPI_Finalize failed\n");
        exit (0);
    }
    return 0;
}
