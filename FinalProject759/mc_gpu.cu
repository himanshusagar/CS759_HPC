#include <iostream>
#include <cstdio>
#include <utility>
#include <cstdlib>

#include "helper_kernels.cuh"

using std::pair;
typedef std::pair<int, int> PII;


static PII get_launch_pair(char c, int n_paths)
{
    PII ret;
    if(c == 'p' || c == 's')
    {
        const int thread_per_block = 256;
        int grid_dim =  (n_paths + thread_per_block - 1) / thread_per_block;
        ret = { grid_dim , thread_per_block };
    }

    return ret;
}

static void gpu_version(const Params &param, double *h_price) 
{
    int grid_dim;
    // HOST: generate random values
    size_t simulation_size = param.n_timestamp * param.n_paths;
    double *host_random_input = gen_host_random_samples(simulation_size , true);

    double *d_paths = NULL;
    cudaMalloc( (void**) &d_paths, param.n_timestamp *  param.n_paths * sizeof(double) ) ;
      
    // The discounted payoffs are the last column.
    double *d_cashflows = d_paths + (param.n_timestamp - 1 ) *  param.n_paths;;

    int max_temp_storage = 4 * 2048;
    double *d_temp_storage = NULL;
    cudaMalloc((void**) &d_temp_storage, max_temp_storage*sizeof(double)) ;
  

    PII ret;
    // Stage 1 : calculated host paths.

    ret = get_launch_pair('p' , param.n_paths  );
    generate_paths_kernel<<< ret.first , ret.second >>>(
        param.n_timestamp,
        param.n_paths,
        param.strike_price, 
        param.dt, 
        param.S0, 
        param.R, 
        param.sigma, 
        host_random_input,
        d_paths);
    CHECK_CUDA(cudaGetLastError());

     // Prepare the SVDs.
    ret = get_launch_pair('s' , param.n_paths );

    int *d_all_out_of_the_money = NULL;
    CHECK_CUDA(cudaMallocManaged( (void**)&d_all_out_of_the_money , param.n_timestamp * sizeof(float) ) );

    double *d_svds = NULL;
    CHECK_CUDA(cudaMallocManaged( (void**)&d_svds, 16 * param.n_timestamp * sizeof(double)));

    const int NUM_THREADS_PER_BLOCK1 = 256;
    prepare_svd_kernel< NUM_THREADS_PER_BLOCK1 ><<< ret.first , ret.second >>>(
        param.n_paths,
        4, 
        param.strike_price, 
        d_paths, 
        d_all_out_of_the_money,
        d_svds);
    CHECK_CUDA(cudaGetLastError());

    // The constant to discount the payoffs.
    const double exp_min_r_dt = std::exp(-param.R * param.dt);

    // hsagar : Wrong Begin
    // Estimate the number of blocks in a wave of update_cashflow.
  cudaDeviceProp properties;
  int device = 0;
   cudaGetDevice(&device) ;
  cudaGetDeviceProperties(&properties, device) ;

  // The number of SMs.
  const int num_sms = properties.multiProcessorCount;
  // Number of threads per wave at fully occupancy.
  const int num_threads_per_wave_full_occupancy = properties.maxThreadsPerMultiProcessor*num_sms;

  // Enable 8B mode for SMEM.
  const int NUM_THREADS_PER_BLOCK2 = 128;

  // Update the cashflows.
  grid_dim = (param.n_paths + NUM_THREADS_PER_BLOCK2-1) / NUM_THREADS_PER_BLOCK2;
  double num_waves = grid_dim*NUM_THREADS_PER_BLOCK2 / (double) num_threads_per_wave_full_occupancy;

  int update_cashflow_grid = grid_dim;
  if( num_waves < 10 && num_waves - (int) num_waves < 0.6 )
    update_cashflow_grid = std::max(1, (int) num_waves) * num_threads_per_wave_full_occupancy / NUM_THREADS_PER_BLOCK2;
    // hsagar : Wrong End

    // Run the main loop.
  for( int timestep = param.n_timestamp - 2 ; timestep >= 0 ; --timestep )
  {
    // Compute beta (two kernels) for that timestep.
    compute_partial_beta_kernel<NUM_THREADS_PER_BLOCK2><<<NUM_THREADS_PER_BLOCK2, NUM_THREADS_PER_BLOCK2>>>(
      param.n_paths,
      param.strike_price,
      d_svds + 16*timestep,
      d_paths + timestep * param.n_paths,
      d_cashflows,
      d_all_out_of_the_money + timestep,
      d_temp_storage);
    CHECK_CUDA(cudaGetLastError());

    compute_final_beta_kernel< NUM_THREADS_PER_BLOCK2 ><<<1, NUM_THREADS_PER_BLOCK2>>>(
      d_all_out_of_the_money + timestep,
      d_temp_storage);
    CHECK_CUDA(cudaGetLastError());

    update_cashflow_kernel< NUM_THREADS_PER_BLOCK2 ><<<update_cashflow_grid, NUM_THREADS_PER_BLOCK2>>>(
      param.n_paths,
      param.strike_price,
      exp_min_r_dt,
      d_temp_storage,
      d_paths + timestep * param.n_paths,
      d_all_out_of_the_money + timestep,
      d_cashflows);

    CHECK_CUDA(cudaGetLastError());
  }

    // Compute the final sum.
    const int NUM_THREADS_PER_BLOCK4 = 128;
    grid_dim = (param.n_paths + NUM_THREADS_PER_BLOCK4-1) / NUM_THREADS_PER_BLOCK4;
    
    compute_partial_sums_kernel< NUM_THREADS_PER_BLOCK4 ><<<grid_dim, NUM_THREADS_PER_BLOCK4>>>(
      param.n_paths,
      d_cashflows,
      d_temp_storage);
    CHECK_CUDA(cudaGetLastError());
  
    compute_final_sum_kernel< NUM_THREADS_PER_BLOCK4 ><<<1, NUM_THREADS_PER_BLOCK4>>>(
      param.n_paths,
      grid_dim,
      exp_min_r_dt,
      d_temp_storage);
    CHECK_CUDA(cudaGetLastError());
  
    // Copy the result to the host.
    cudaMemcpy(h_price, d_temp_storage, sizeof(double), cudaMemcpyDeviceToHost);

}



int main(int argc, char **argv) 
{

    double T = 1.00;

    Params param;
    param.n_timestamp = 100;
    param.n_paths = 32 * 1024;
    param.S0 = 3.60;
    param.strike_price = 4.00;
    param.dt = T / param.n_timestamp;
    param.R = 0.06;;
    param.sigma = 0.20;;
    param.pretty_print();

    // The price on the host.
  double *h_price = NULL;
  cudaHostAlloc((void**) &h_price, sizeof(double), cudaHostAllocDefault);

    float time = 0;
    {
        UnitGPUTime g;
        gpu_version(param, h_price);
        time = g.getTime();
        printf("GPU Longstaff-Schwartz: %.8lf\n", *h_price);
    }
    printf("%f\n", time);

    return 0;
}
