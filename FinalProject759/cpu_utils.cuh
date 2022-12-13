#include "utils.cuh"

class Params
{
public:
  int n_timestamp; 
  int n_paths;
  double S0; 
  double R; 
  double sigma;
  bool with_debug_info;
  double strike_price; 
  double dt; 

  void pretty_print()
  {
    printf("==============\n");
    printf("Num Timesteps         : %d\n",  n_timestamp);
    printf("Num Paths             : %dK\n", n_paths / 1024);
    printf("S0                    : %lf\n", S0);
    printf("K                     : %lf\n", strike_price);
    printf("R                     : %lf\n", R);
    printf("sigma                 : %lf\n", sigma);
  }
};

static double* gen_host_paths(const Params& param, const double *host_random_input)
{
    size_t simulation_size = param.n_timestamp * param.n_paths;

    double *host_paths = new double[simulation_size];
    const double A = (param.R - 0.5f * param.sigma * param.sigma) * param.dt;
    const double B = param.sigma * sqrt(param.dt);

    int i_timestamp = 0;
    while(i_timestamp < param.n_timestamp)
    {
        int i_path = 0;
        while( i_path < param.n_paths )
        {
            double S = 0;
            if(i_timestamp == 0)
                S = param.S0;
            else 
                S = host_paths[ (i_timestamp - 1) * param.n_paths + i_path ];
                
            S = S * exp( A + B * host_random_input[ i_timestamp * param.n_paths + i_path ] );

            if(i_timestamp < param.n_timestamp - 1)
                host_paths[ i_timestamp * param.n_paths + i_path] = S;
            else
                host_paths[ i_timestamp * param.n_paths + i_path] = payOffOverS(S , param.strike_price);

            i_path++;
        }
        i_timestamp++;
    }
    return host_paths;

}