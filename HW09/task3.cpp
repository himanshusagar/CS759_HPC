#include <iostream>
#include <algorithm>    
#include <chrono>
#include <string>
#include <cmath>
#include <cstring>

#include "mpi.h"

#include "profile.h"

int main(int argc, char *argv[])
{
    int my_rank;       /* rank of process      */
    int p;             /* number of processes  */
    int source;        /* rank of sender       */
    int dest;          /* rank of receiver     */
    int tag = 0;       /* tag for messages     */
    MPI_Status status;        /* return status for receive  */

    MPI_Init(&argc, &argv); // Start up MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank); // Find out process rank
    MPI_Comm_size(MPI_COMM_WORLD, &p); // Find out number of processes

    size_t arr_size = std::stoi(argv[1]);
    float *send_buffer = new float[arr_size];
    float *recv_buffer = new float[arr_size];

    // Generate random values
    std::default_random_engine e;
    e.seed(std::chrono::system_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<> dis(-5, 5);

    //Fill buffers with random values
    for (size_t i = 0; i < arr_size; i++)
    {
        send_buffer[i] = dis(e);
        recv_buffer[i] = dis(e);
    }

    if( my_rank == 0) 
    {
        dest = 1;
        source = 1;
        //zero process - main 
        float time_taken;
        {
            UnitTime u;
            //Send and Recv into buffers.
            MPI_Send(send_buffer, arr_size, MPI_FLOAT, dest, tag + 1, MPI_COMM_WORLD);
            MPI_Recv(recv_buffer, arr_size, MPI_FLOAT, source, tag + 2, MPI_COMM_WORLD, &status);
            time_taken = u.getTime();
            //Time calculation
           
        }
        float their_time_taken;
        MPI_Recv( &their_time_taken, 1, MPI_FLOAT, source, tag + 3, MPI_COMM_WORLD, &status);
        float tot_val = time_taken + their_time_taken ;
        //MPI total time.
        std::cout << tot_val <<  std::endl;

    }
    else
    {
        dest = 0;
        source = 0;
        float time_taken;
        {
            UnitTime u;
            //Recv and Send into buffers.
            MPI_Recv(send_buffer, arr_size, MPI_FLOAT, source, tag + 1, MPI_COMM_WORLD, &status); 
            MPI_Send(recv_buffer, arr_size, MPI_FLOAT, dest, tag + 2, MPI_COMM_WORLD);
            time_taken = u.getTime();
        }

        // Time to send time taken;
        dest = 0;
        MPI_Send( &time_taken, 1 , MPI_FLOAT, dest, tag + 3, MPI_COMM_WORLD);
    }
    return 0;
}
