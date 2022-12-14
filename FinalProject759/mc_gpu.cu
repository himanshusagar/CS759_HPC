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
    if(c == 's' || c == 'p')
    {
        float f_n_paths = n_paths;
        int grid_dim =  std::ceil(f_n_paths / 256.0);
        ret = { grid_dim , 256 };
    }
    else if(c == 'q')
    {
        float f_n_paths = n_paths;
        int grid_dim =  std::ceil(f_n_paths / 128.0);
        ret = { grid_dim , 128 };
    }

    return ret;
}

static void gpu_version(const Params &param, double *h_price) 
{

    size_t simulation_size = param.n_timestamp * param.n_paths;
    double *d_rand_samples = gen_host_random_samples(simulation_size , true);

    double *d_sim_paths = NULL;
    cudaMalloc( (void**) &d_sim_paths, param.n_timestamp *  param.n_paths * sizeof(double) ) ;
      
    double *d_cash = d_sim_paths + (param.n_timestamp - 1 ) *  param.n_paths;;

    double *d_tmp = NULL;
    cudaMalloc((void**) &d_tmp, 4 * 2048 * sizeof(double)) ;
  

    PII ret;
    ret = get_launch_pair('p' , param.n_paths  );
    step_1_paths_kernel<<< ret.first , ret.second >>>(
        param.n_timestamp,
        param.n_paths,
        param.strike_price, 
        param.dt, 
        param.S0, 
        param.R, 
        param.sigma, 
        d_rand_samples,
        d_sim_paths);
    CHECK_CUDA(cudaGetLastError());

    ret = get_launch_pair('s' , param.n_paths );

    int *d_no_money = NULL;
    CHECK_CUDA(cudaMallocManaged( (void**)&d_no_money , param.n_timestamp * sizeof(float) ) );

    CHECK_CUDA(cudaMemsetAsync(d_no_money, 0, param.n_timestamp * sizeof(int)));
 
    double *d_svds = NULL;
    CHECK_CUDA(cudaMallocManaged( (void**)&d_svds, 16 * param.n_timestamp * sizeof(double)));

    stage_2_svd_kernel<<< param.n_timestamp-1 , 256 >>>(
        param.n_paths,
        4, 
        param.strike_price, 
        d_sim_paths, 
        d_no_money,
        d_svds);
    CHECK_CUDA(cudaGetLastError());

    const double alpha_param = std::exp(-param.R * param.dt);


  ret = get_launch_pair('q' , param.n_paths  );

  for( int timestep = param.n_timestamp - 2 ; timestep >= 0 ; --timestep )
  {
    stage_5_p_beta_kernel<<< ret.second , ret.second>>>(
      param.n_paths,
      param.strike_price,
      d_svds + 16*timestep,
      d_sim_paths + timestep * param.n_paths,
      d_cash,
      d_no_money + timestep,
      d_tmp);

    stage_6_f_beta_kernel<<< 1, ret.second >>>(
      d_no_money + timestep,
      d_tmp);
      
      move_cash_kernel<<< ret.first, ret.second >>>(
      param.n_paths,
      param.strike_price,
      alpha_param,
      d_tmp,
      d_sim_paths + timestep * param.n_paths,
      d_no_money + timestep,
      d_cash);

    CHECK_CUDA(cudaGetLastError());
  }

    ret = get_launch_pair('q' , param.n_paths  );
    stage_3_p_sum_kernel<<<ret.first , ret.second>>>(
      param.n_paths,
      d_cash,
      d_tmp);

    stage_4_f_sum_kernel<<<1, ret.second>>>(
      param.n_paths,
      ret.first,
      alpha_param,
      d_tmp);
  
    cudaMemcpy(h_price, d_tmp, sizeof(double), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
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
    param.R = 0.06;
    param.sigma = 0.2;
    param.pretty_print();

    // The price on the host.
  double *h_price = NULL;
  cudaHostAlloc((void**) &h_price, sizeof(double), cudaHostAllocDefault);

    float time = 0;
    //std::cout << timestamp << ", ";
    {
        UnitGPUTime g;
        gpu_version(param, h_price);
        time = g.getTime();
        printf("GPU Longstaff-Schwartz: %.8lf\n", *h_price);
    }
    cout << time << ", " << endl;

    return 0;
}
