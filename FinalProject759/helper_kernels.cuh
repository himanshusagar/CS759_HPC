#ifndef HELPER_KERNELS_H
#define HELPER_KERNELS_H

#include <cuda_runtime_api.h>
#include "utils.cuh"
#include <cub/cub.cuh>


typedef cub::BlockScan< int, 128 > BlockScan128;
typedef cub::BlockReduce<double, 128> BlockReduce128;
typedef cub::BlockReduce<double3, 128> BlockReduce3_128;
  
typedef cub::BlockScan< int, 256 > BlockScan256;
typedef cub::BlockReduce<int, 256 > BlockReduce256;
typedef cub::BlockReduce<double4, 256 > BlockReduce4_256;


__global__  void stage_5_p_beta_kernel(int num_paths,
                                             double strike_price,
                                             const double * svd,
                                             const double * paths,
                                             const double * cashflows,
                                             const int * no_money,
                                             double * p_tot_sums)
{

    __shared__ typename BlockReduce3_128::TempStorage smem_for_reduce;

    __shared__ double shared_svd[12];

    const int NUM_THREADS_PER_GRID = 128 * gridDim.x;

    if( threadIdx.x < 12 )
    {
        shared_svd[threadIdx.x] = svd[threadIdx.x];
    }
    __syncthreads();
    
    const double R00 = shared_svd[ 0];
    const double R01 = shared_svd[ 1];
    const double R02 = shared_svd[ 2];
    const double R11 = shared_svd[ 3];
    const double R12 = shared_svd[ 4];
    const double R22 = shared_svd[ 5];

    const double W00 = shared_svd[ 6];
    const double W01 = shared_svd[ 7];
    const double W02 = shared_svd[ 8];
    const double W11 = shared_svd[ 9];
    const double W12 = shared_svd[10];
    const double W22 = shared_svd[11];

    
    const double inv_R00 = R00 != 0.0 ? __drcp_rn(R00) : 0.0;
    const double inv_R11 = R11 != 0.0 ? __drcp_rn(R11) : 0.0;
    const double inv_R22 = R22 != 0.0 ? __drcp_rn(R22) : 0.0;

    
    const double inv_R01 = inv_R00*inv_R11*R01;
    const double inv_R02 = inv_R00*inv_R22*R02;
    const double inv_R12 =         inv_R22*R12;


    const double inv_W00 = W00*inv_R00;

    
    double beta0 = 0.0, beta1 = 0.0, beta2 = 0.0;

    
    for( int path = blockIdx.x*128 + threadIdx.x ; path < num_paths ; path += NUM_THREADS_PER_GRID )
    {
        
        double S = paths[path];

        
        const int good_money = S > strike_price;

        
        double Q1i = inv_R11*S - inv_R01;
        double Q2i = inv_R22*S*S - inv_R02 - Q1i*inv_R12;


        const double WI0 = inv_W00 + W01 * Q1i + W02 * Q2i;
        const double WI1 =           W11 * Q1i + W12 * Q2i;
        const double WI2 =                       W22 * Q2i;

        double cashflow = good_money ? cashflows[path] : 0.0;

        beta0 += WI0*cashflow;
        beta1 += WI1*cashflow;
        beta2 += WI2*cashflow;
    }

    double3 tot_sums = BlockReduce3_128(smem_for_reduce).Sum(make_double3(beta0, beta1, beta2));


    if( threadIdx.x == 0 )
    {
        p_tot_sums[0*128 + blockIdx.x] = tot_sums.x;
        p_tot_sums[1*128 + blockIdx.x] = tot_sums.y;
        p_tot_sums[2*128 + blockIdx.x] = tot_sums.z;
    }

}


__host__ __device__ double3 operator+(const double3 &u, const double3 &v )
{
  return make_double3(u.x+v.x, u.y+v.y, u.z+v.z);
}

__host__ __device__ double4 operator+(const double4 &u, const double4 &v )
{
  return make_double4(u.x+v.x, u.y+v.y, u.z+v.z, u.w+v.w);
}

__global__ void step_1_paths_kernel(int n_timestamp,
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



__global__ void stage_2_svd_kernel(int n_paths,
                                   int min_in_the_money,
                                   double strike_price,
                                   const double *paths,
                                   int *no_money,
                                   double *svds) {
    int tIndex = threadIdx.x;
    union TempStorage {
        typename BlockScan256::TempStorage for_scan;
        typename BlockReduce256::TempStorage for_reduce1;
        typename BlockReduce4_256::TempStorage for_reduce4;
    };
    __shared__
    TempStorage smem_storage;
    __shared__ double s_mem_svd[12];

    int tot_money = 0;
    double4 tot_sums = {0.0, 0.0, 0.0, 0.0};

    int found_paths = 0;
    int path = tIndex;
    while (path < n_paths) {
        double S = 0.0;
        if (path < n_paths) {
            S = paths[blockIdx.x * n_paths + path];
        }

        const int good_money = S > strike_price;

        if (found_paths < 3) {
            int partial_sum = 0, total_sum = 0;
            BlockScan256(smem_storage.for_scan).ExclusiveSum(good_money, partial_sum, total_sum);

            if (good_money && found_paths + partial_sum < 3) {
                s_mem_svd[found_paths + partial_sum] = S;
            }
            __syncthreads();
            found_paths += total_sum;
        }

        tot_money += good_money;

        double x = 0.0, x2 = 0.0;
        if (good_money) {
            x = S;
            x2 = S * S;
        }


        tot_sums.x += x;
        tot_sums.y += x2;
        tot_sums.z += x2 * x;
        tot_sums.w += x2 * x2;

        path += 256;
    }

    __syncthreads();

    tot_money = BlockReduce256(smem_storage.for_reduce1).Sum(tot_money);

    int shouldExit = __syncthreads_or(tIndex == 0 && tot_money < min_in_the_money);

    if (shouldExit) {
        if (tIndex == 0)
            no_money[blockIdx.x] = 1;
        return;
    }


    tot_sums = BlockReduce4_256(smem_storage.for_reduce4).Sum(tot_sums);

    if (tIndex == 0) {
        svd_3x3(tot_money, tot_sums, s_mem_svd);
    }
    __syncthreads();
    if (tIndex < 12)
        svds[16 * blockIdx.x + tIndex] = s_mem_svd[tIndex];
}



__global__  void stage_3_p_sum_kernel(int num_paths, const double *__restrict cashflows, double *__restrict sums)
{
  __shared__ typename BlockReduce128::TempStorage smem_storage;

  const int path = blockIdx.x * 128 + threadIdx.x;

  double sum = 0.0;
  if( path < num_paths )
  {
    sum = cashflows[path];
  }

  sum = BlockReduce128(smem_storage).Sum(sum);
  if( threadIdx.x == 0 )
  {
    sums[blockIdx.x] = sum;
  }
}


__global__  void move_cash_kernel(int num_paths,
                                        double strike_price,
                                        double alpha_val,
                                        const double * beta,
                                        const double * paths,
                                        const int * no_money,
                                        double * cashflows)
{
    const int NUM_THREADS_PER_GRID = gridDim.x * 128;
    const int no_comp = *no_money;
    const double beta0 = beta[0];
    const double beta1 = beta[1];
    const double beta2 = beta[2];

    int path = blockIdx.x*128 + threadIdx.x;
    for( ; path < num_paths ; path += NUM_THREADS_PER_GRID )
    {
        const double old_cashflow = alpha_val*cashflows[path];
        if( no_comp )
        {
            cashflows[path] = old_cashflow;
            continue;
        }
        double S  = paths[path];
        double S2 = S*S;
        double payoff = payOffOverS(S , strike_price);
        double estimated_payoff = beta0 + beta1*S + beta2*S2;
        estimated_payoff *= alpha_val;
        if( payoff <= 1.0e-8 || payoff <= estimated_payoff )
            payoff = old_cashflow;
        cashflows[path] = payoff;
    }
}

__global__ void stage_6_f_beta_kernel(const int *__restrict no_money, double *__restrict beta)
{

  __shared__ typename BlockReduce3_128::TempStorage smem_for_reduce;

  if( *no_money )
  {
    if( threadIdx.x < 3 )
      beta[threadIdx.x] = 0.0;
    return;
  }

  double3 sums;
  
  sums.x = beta[0*128 + threadIdx.x];
  sums.y = beta[1*128 + threadIdx.x];
  sums.z = beta[2*128 + threadIdx.x];
  
  sums = BlockReduce3_128(smem_for_reduce).Sum(sums);

  if( threadIdx.x == 0 )
  {
    beta[0] = sums.x; 
    beta[1] = sums.y;
    beta[2] = sums.z;
  }
}


__global__ void stage_4_f_sum_kernel(int num_paths, int num_blocks, double alpha, double * sums)
{
  __shared__ typename BlockReduce128::TempStorage smem_storage;
  double sum = 0.0;
  for( int item = threadIdx.x ; item < num_blocks ; item += 128 )
  {
      sum += sums[item];
  }
  sum = BlockReduce128(smem_storage).Sum(sum);

  if( threadIdx.x == 0 )
  {
    sums[0] = alpha * sum / (double) num_paths;
  }
}

#endif //HELPER_KERNELS_H