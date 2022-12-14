#ifndef HELPER_KERNELS_H
#define HELPER_KERNELS_H

#include <cuda_runtime_api.h>
#include "utils.cuh"
#include <cub/cub.cuh>

#define R_W_MATRICES_SMEM_SLOTS 12
#define HOST_DEVICE        

__host__ __device__ double3 operator+(const double3 &u, const double3 &v )
{
  return make_double3(u.x+v.x, u.y+v.y, u.z+v.z);
}

__host__ __device__ double4 operator+(const double4 &u, const double4 &v )
{
  return make_double4(u.x+v.x, u.y+v.y, u.z+v.z, u.w+v.w);
}

__global__ void generate_paths_kernel(int n_timestamp,
                                      int n_paths,
                                      double strike_price,
                                      double dt,
                                      double S0,
                                      double r,
                                      double sigma,
                                      const double *samples,
                                      double *paths) {
    int path = blockIdx.x * 256 + threadIdx.x;

    if (path >= n_paths)
        return;

    const double A = (r - 0.5 * sigma * sigma) * dt;
    const double B = sigma * sqrt(dt);

    double S = S0;
    int offset = path;
    int i_timestamp = 0;
    while (i_timestamp < n_timestamp) {
        S = S * exp(A + B * samples[offset]);

        if (i_timestamp < n_timestamp - 1)
            paths[offset] = S;
        else
            paths[offset] = payOffOverS(S, strike_price);

        i_timestamp++;
        offset += n_paths;
    }
}

template< int NUM_THREADS_PER_BLOCK >
__global__ void prepare_svd_kernel(int num_paths,
                        int min_in_the_money,
                        double strike_price,
                        const double*  paths,
                        int* all_out_of_the_money,
                        double* svds) 
                        {
// We need to perform a scan to find the first 3 stocks pay off.
    typedef cub::BlockScan< int, NUM_THREADS_PER_BLOCK > BlockScan;

// We need to perform a reduction at the end of the kernel to compute the final sums.
    typedef cub::BlockReduce<int, NUM_THREADS_PER_BLOCK > BlockReduce1;
    typedef cub::BlockReduce<double4, NUM_THREADS_PER_BLOCK > BlockReduce4;

// The union for the scan/reduce.
    union TempStorage {
        typename BlockScan::TempStorage for_scan;
        typename BlockReduce1::TempStorage for_reduce1;
        typename BlockReduce4::TempStorage for_reduce4;
    };

// Shared memory.
    __shared__
    TempStorage smem_storage;

// Shared buffer for the ouput.
    __shared__ double smem_svds[R_W_MATRICES_SMEM_SLOTS];

// Each block works on a single timestep. 
    const int timestep = blockIdx.x;
// The timestep offset.
    const int offset = timestep * num_paths;

// Sums.
    int m = 0;
    double4 sums = {0.0, 0.0, 0.0, 0.0};

// Initialize the shared memory. DBL_MAX is a marker to specify that the value is invalid.
    if (threadIdx.x < R_W_MATRICES_SMEM_SLOTS)
        smem_svds[threadIdx.x] = 0.0;
    __syncthreads();

// Have we already found our 3 first paths which pay off.
    int found_paths = 0;

// Iterate over the paths.
    for (int path = threadIdx.x; path < num_paths; path += NUM_THREADS_PER_BLOCK) {
// Load the asset price to determine if it pays off.
        double S = 0.0;
        if (path < num_paths)
            S = paths[offset + path];

// Check if it pays off.
        const int in_the_money = S > strike_price;

// Try to check if we have found the 3 first stocks.
        if (found_paths < 3) {
            int partial_sum = 0, total_sum = 0;
            BlockScan(smem_storage.for_scan).ExclusiveSum(in_the_money, partial_sum, total_sum);
            if (in_the_money && found_paths + partial_sum < 3)
                smem_svds[found_paths + partial_sum] = S;
            __syncthreads();
            found_paths += total_sum;
        }

// Update the number of payoff items.
        m += in_the_money;

// The "normalized" value.
        double x = 0.0, x_sq = 0.0;
        if (in_the_money) {
            x = S;
            x_sq = S * S;
        }

// Compute the 4 sums.
        sums.x += x;
        sums.y += x_sq;
        sums.z += x_sq * x;
        sums.w += x_sq * x_sq;
    }

// Make sure the scan is finished.
    __syncthreads();

// Compute the final reductions.
    m = BlockReduce1(smem_storage.for_reduce1).Sum(m);

// Do we all exit?
    int not_enough_paths = __syncthreads_or(threadIdx.x == 0 && m < min_in_the_money);

// Early exit if no path is in the money.
    if (not_enough_paths) {
        if (threadIdx.x == 0)
            all_out_of_the_money[blockIdx.x] = 1;
        return;
    }

// Compute the final reductions.
    sums = BlockReduce4(smem_storage.for_reduce4).Sum(sums);

// The 1st thread has everything he needs to build R from the QR decomposition.
    if (threadIdx.x == 0)
        svd_3x3(m, sums, smem_svds);
    __syncthreads();

// Store the final results.
    if (threadIdx.x < R_W_MATRICES_SMEM_SLOTS)
        svds[16 * blockIdx.x + threadIdx.x] = smem_svds[threadIdx.x];
}



template< int NUM_THREADS_PER_BLOCK >
__global__ void compute_final_sum_kernel(int num_paths, int num_blocks, double exp_min_r_dt, double *__restrict sums)
{
  typedef cub::BlockReduce<double, NUM_THREADS_PER_BLOCK > BlockReduce;

  // Shared memory to compute the final sum.
  __shared__ typename BlockReduce::TempStorage smem_storage;

  // The sum.
  double sum = 0.0;
  for( int item = threadIdx.x ; item < num_blocks ; item += NUM_THREADS_PER_BLOCK )
    sum += sums[item];

  // Compute the sum over the block.
  sum = BlockReduce(smem_storage).Sum(sum);

  // The block leader writes the sum to GMEM.
  if( threadIdx.x == 0 )
  {
    sums[0] = exp_min_r_dt * sum / (double) num_paths;
  }
}

template< int NUM_THREADS_PER_BLOCK >
__global__  void compute_partial_sums_kernel(int num_paths, const double *__restrict cashflows, double *__restrict sums)
{
  typedef cub::BlockReduce<double, NUM_THREADS_PER_BLOCK> BlockReduce;

  // Shared memory to compute the final sum.
  __shared__ typename BlockReduce::TempStorage smem_storage;

  // Each thread works on a single path.
  const int path = blockIdx.x * NUM_THREADS_PER_BLOCK + threadIdx.x;

  // Load the final sum.
  double sum = 0.0;
  if( path < num_paths )
    sum = cashflows[path];

  // Compute the sum over the block.
  sum = BlockReduce(smem_storage).Sum(sum);

  // The block leader writes the sum to GMEM.
  if( threadIdx.x == 0 )
    sums[blockIdx.x] = sum;
}

template< int NUM_THREADS_PER_BLOCK >
__global__  void update_cashflow_kernel(int num_paths,
                            double strike_price,
                            double exp_min_r_dt,
                            const double * beta,
                            const double * paths,
                            const int * all_out_of_the_money,
                            double * cashflows)
{
  const int NUM_THREADS_PER_GRID = gridDim.x * NUM_THREADS_PER_BLOCK;

  // Are we going to skip the computations.
  const int skip_computations = *all_out_of_the_money;

#ifdef WITH_FUSED_BETA
  typedef cub::BlockReduce<double3, NUM_THREADS_PER_BLOCK> BlockReduce;

  // The shared memory for the reduction.
  __shared__ typename BlockReduce::TempStorage smem_for_reduce;
  // The shared memory to exchange beta.
  __shared__ double smem_beta[3];

  // The final sums.
  double3 sums;
  
  // We load the elements. Each block loads the same elements.
  sums.x = beta[0*NUM_THREADS_PER_BLOCK + threadIdx.x];
  sums.y = beta[1*NUM_THREADS_PER_BLOCK + threadIdx.x];
  sums.z = beta[2*NUM_THREADS_PER_BLOCK + threadIdx.x];
  
  // Compute the sums.
  sums = BlockReduce(smem_for_reduce).Sum(sums);

  // Store beta.
  if( threadIdx.x == 0 )
  {
    smem_beta[0] = sums.x; 
    smem_beta[1] = sums.y;
    smem_beta[2] = sums.z;
  }
  __syncthreads();

  // Load the beta coefficients from SMEM.
  const double beta0 = smem_beta[0];
  const double beta1 = smem_beta[1];
  const double beta2 = smem_beta[2];
#else
  // Load the beta coefficients for the linear regression.
  const double beta0 = beta[0];
  const double beta1 = beta[1];
  const double beta2 = beta[2];
#endif

  // Iterate over the paths.
  int path = blockIdx.x*NUM_THREADS_PER_BLOCK + threadIdx.x;
  for( ; path < num_paths ; path += NUM_THREADS_PER_GRID )
  {
    // The cashflow.
    const double old_cashflow = exp_min_r_dt*cashflows[path];
    if( skip_computations )
    {
      cashflows[path] = old_cashflow;
      continue;
    }
  
    // Load the asset price.
    double S  = paths[path];
    double S2 = S*S;

    // The payoff
    double payoff = payOffOverS(S , strike_price);

    // Compute the estimated payoff from continuing.
    double estimated_payoff = beta0 + beta1*S + beta2*S2;

    // Discount the payoff because we did not take it into account for beta.
    estimated_payoff *= exp_min_r_dt;

    // Update the payoff
    if( payoff <= 1.0e-8 || payoff <= estimated_payoff )
      payoff = old_cashflow;
    
    // Store the updated cashflow.
    cashflows[path] = payoff;
  }
}

template< int NUM_THREADS_PER_BLOCK >
__global__ void compute_final_beta_kernel(const int *__restrict all_out_of_the_money, double *__restrict beta)
{
  typedef cub::BlockReduce<double3, NUM_THREADS_PER_BLOCK> BlockReduce;

  // The shared memory for the reduction.
  __shared__ typename BlockReduce::TempStorage smem_for_reduce;

  // Early exit if needed.
  if( *all_out_of_the_money )
  {
    if( threadIdx.x < 3 )
      beta[threadIdx.x] = 0.0;
    return;
  }

  // The final sums.
  double3 sums;
  
  // We load the elements.
  sums.x = beta[0*NUM_THREADS_PER_BLOCK + threadIdx.x];
  sums.y = beta[1*NUM_THREADS_PER_BLOCK + threadIdx.x];
  sums.z = beta[2*NUM_THREADS_PER_BLOCK + threadIdx.x];
  
  // Compute the sums.
  sums = BlockReduce(smem_for_reduce).Sum(sums);

  // Store beta.
  if( threadIdx.x == 0 )
  {
    //printf("beta0=%.8lf beta1=%.8lf beta2=%.8lf\n", sums.x, sums.y, sums.z);
    beta[0] = sums.x; 
    beta[1] = sums.y;
    beta[2] = sums.z;
  }
}

template< int NUM_THREADS_PER_BLOCK >
__global__  void compute_partial_beta_kernel(int num_paths,
                                 double strike_price,
                                 const double * svd,
                                 const double * paths,
                                 const double * cashflows,
                                 const int * all_out_of_the_money,
                                 double * partial_sums)
{
  typedef cub::BlockReduce<double3, NUM_THREADS_PER_BLOCK> BlockReduce;
  
  // The shared memory storage.
  __shared__ typename BlockReduce::TempStorage smem_for_reduce;
  
  // The shared memory to store the SVD.
  __shared__ double shared_svd[R_W_MATRICES_SMEM_SLOTS];
    
  // Early exit if needed.
  if( *all_out_of_the_money )
  {
    return;
  }

  // The number of threads per grid.
  const int NUM_THREADS_PER_GRID = NUM_THREADS_PER_BLOCK * gridDim.x;

  // The 1st threads loads the matrices SVD and R.
  if( threadIdx.x < R_W_MATRICES_SMEM_SLOTS )
    shared_svd[threadIdx.x] = svd[threadIdx.x];
  __syncthreads();

  // Load the terms of R.
  const double R00 = shared_svd[ 0];
  const double R01 = shared_svd[ 1];
  const double R02 = shared_svd[ 2];
  const double R11 = shared_svd[ 3];
  const double R12 = shared_svd[ 4];
  const double R22 = shared_svd[ 5];

  // Load the elements of W.
#ifdef WITH_FULL_W_MATRIX
  const double W00 = shared_svd[ 6];
  const double W01 = shared_svd[ 7];
  const double W02 = shared_svd[ 8];
  const double W10 = shared_svd[ 9];
  const double W11 = shared_svd[10];
  const double W12 = shared_svd[11];
  const double W20 = shared_svd[12];
  const double W21 = shared_svd[13];
  const double W22 = shared_svd[14];
#else
  const double W00 = shared_svd[ 6];
  const double W01 = shared_svd[ 7];
  const double W02 = shared_svd[ 8];
  const double W11 = shared_svd[ 9];
  const double W12 = shared_svd[10];
  const double W22 = shared_svd[11];
#endif

  // Invert the diagonal of R.
  const double inv_R00 = R00 != 0.0 ? __drcp_rn(R00) : 0.0;
  const double inv_R11 = R11 != 0.0 ? __drcp_rn(R11) : 0.0;
  const double inv_R22 = R22 != 0.0 ? __drcp_rn(R22) : 0.0;

  // Precompute the R terms.
  const double inv_R01 = inv_R00*inv_R11*R01;
  const double inv_R02 = inv_R00*inv_R22*R02;
  const double inv_R12 =         inv_R22*R12;
  
  // Precompute W00/R00.
#ifdef WITH_FULL_W_MATRIX
  const double inv_W00 = W00*inv_R00;
  const double inv_W10 = W10*inv_R00;
  const double inv_W20 = W20*inv_R00;
#else
  const double inv_W00 = W00*inv_R00;
#endif

  // Each thread has 3 numbers to sum.
  double beta0 = 0.0, beta1 = 0.0, beta2 = 0.0;

  // Iterate over the paths.
  for( int path = blockIdx.x*NUM_THREADS_PER_BLOCK + threadIdx.x ; path < num_paths ; path += NUM_THREADS_PER_GRID )
  {
    // Threads load the asset price to rebuild Q from the QR decomposition.
    double S = paths[path];

    // Is the path in the money?
    const int in_the_money = S > strike_price;

    // Compute Qis. The elements of the Q matrix in the QR decomposition.
    double Q1i = inv_R11*S - inv_R01;
    double Q2i = inv_R22*S*S - inv_R02 - Q1i*inv_R12;

    // Compute the ith row of the pseudo-inverse of [1 X X^2].
#ifdef WITH_FULL_W_MATRIX
    const double WI0 = inv_W00 + W01 * Q1i + W02 * Q2i;
    const double WI1 = inv_W10 + W11 * Q1i + W12 * Q2i;
    const double WI2 = inv_W20 + W21 * Q1i + W22 * Q2i;
#else
    const double WI0 = inv_W00 + W01 * Q1i + W02 * Q2i;
    const double WI1 =           W11 * Q1i + W12 * Q2i;
    const double WI2 =                       W22 * Q2i;
#endif

    // Each thread loads its element from the Y vector.
    double cashflow = in_the_money ? cashflows[path] : 0.0;
  
    // Update beta.
    beta0 += WI0*cashflow;
    beta1 += WI1*cashflow;
    beta2 += WI2*cashflow;
  }

  // Compute the sum of the elements in the block. We could do slightly better by removing the bank conflicts here.
  double3 sums = BlockReduce(smem_for_reduce).Sum(make_double3(beta0, beta1, beta2));
  
  // The 1st thread stores the result to GMEM.
#ifdef WITH_ATOMIC_BETA
  if( threadIdx.x == 0 )
  {
    atomic_add(&partial_sums[0], sums.x);
    atomic_add(&partial_sums[1], sums.y);
    atomic_add(&partial_sums[2], sums.z);
  }
#else
  if( threadIdx.x == 0 )
  {
    partial_sums[0*NUM_THREADS_PER_BLOCK + blockIdx.x] = sums.x;
    partial_sums[1*NUM_THREADS_PER_BLOCK + blockIdx.x] = sums.y;
    partial_sums[2*NUM_THREADS_PER_BLOCK + blockIdx.x] = sums.z;
  }
#endif
}


#endif //HELPER_KERNELS_H