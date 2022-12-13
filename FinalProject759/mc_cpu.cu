#include <iostream>
#include <cstdio>
#include <cstdlib>

#include "cpu_utils.cuh"

//LAPACK : Weird way to call its functions
extern "C" void
dgesvd_(char *, char *, long *, long *, double *, long *, double *, double *, long *, double *, long *, double *,
        long *, long *);

static double isEarnMoney(double value, const double strike_price) {
    return value > strike_price;
}

static double longstaff_schwartz_cpu(const Params &param) {
    // HOST: generate random values
    size_t simulation_size = param.n_timestamp * param.n_paths;
    double *host_random_input = gen_host_random_samples(simulation_size);

    // HOST: calculated paths.
    double *host_paths = gen_host_paths(param, host_random_input);

    double *h_cashflows = &host_paths[(param.n_timestamp - 1) * param.n_paths];

    const double exp_min_r_dt = std::exp(-param.R * param.dt);

    double *h_matrix = new double[3 * param.n_paths];
    double *h_S = new double[3];
    double *h_U = new double[3 * param.n_paths];
    double *h_V = new double[3 * 3];
    double *h_work = new double[param.n_paths + 3 * 3];

    for (int timestep = param.n_timestamp - 2; timestep >= 0; --timestep) {
        long m = 0;

        for (int i = 0; i < param.n_paths; ++i) {
            double S = host_paths[timestep * param.n_paths + i];
            if (!isEarnMoney(S, param.strike_price))
                continue;

            h_matrix[0 * param.n_paths + m] = 1.0;
            h_matrix[1 * param.n_paths + m] = S;
            h_matrix[2 * param.n_paths + m] = S * S;

            m++;
        }

        char JOBU = 'S', JOBVT = 'S';
        long ldm = param.n_paths;
        long N = 3;
        long LWORK = param.n_paths + 3 * 3;
        long info = 0;
        dgesvd_(&JOBU, &JOBVT, &m, &N, h_matrix, &ldm, h_S, h_U, &ldm, h_V, &N, h_work, &LWORK, &info);
        if (info) {
            fprintf(stderr, "LAPACK error at line %d: %ld\n", __LINE__, info);
            exit(1);
        }

        double inv_S0 = abs(h_S[0]) < 1.0e-12 ? 0.0 : 1.0 / h_S[0];
        double inv_S1 = abs(h_S[1]) < 1.0e-12 ? 0.0 : 1.0 / h_S[1];
        double inv_S2 = abs(h_S[2]) < 1.0e-12 ? 0.0 : 1.0 / h_S[2];

        h_V[0] *= inv_S0;
        h_V[1] *= inv_S1;
        h_V[2] *= inv_S2;
        h_V[3] *= inv_S0;
        h_V[4] *= inv_S1;
        h_V[5] *= inv_S2;
        h_V[6] *= inv_S0;
        h_V[7] *= inv_S1;
        h_V[8] *= inv_S2;

        for (int i = 0; i < m; ++i) {
            double a = h_U[0 * param.n_paths + i];
            double b = h_U[1 * param.n_paths + i];
            double c = h_U[2 * param.n_paths + i];

            h_U[0 * param.n_paths + i] = a * h_V[0] + b * h_V[1] + c * h_V[2];
            h_U[1 * param.n_paths + i] = a * h_V[3] + b * h_V[4] + c * h_V[5];
            h_U[2 * param.n_paths + i] = a * h_V[6] + b * h_V[7] + c * h_V[8];
        }

        double beta0 = 0.0, beta1 = 0.0, beta2 = 0.0;
        for (int i = 0, k = 0; i < param.n_paths; ++i) {
            double S = host_paths[timestep * param.n_paths + i];
            if (!isEarnMoney(S, param.strike_price))
                continue;

            double cashflow = h_cashflows[i];

            beta0 += h_U[0 * param.n_paths + k] * cashflow;
            beta1 += h_U[1 * param.n_paths + k] * cashflow;
            beta2 += h_U[2 * param.n_paths + k] * cashflow;

            k++;
        }

        for (int i = 0; i < param.n_paths; ++i) {
            double S = host_paths[timestep * param.n_paths + i];
            double p = payOffOverS(S, param.strike_price);

            double estimated_payoff = exp_min_r_dt * (beta0 + beta1 * S + beta2 * S * S);

            if (p <= 1.0e-8 || p <= estimated_payoff)
                p = exp_min_r_dt * h_cashflows[i];
            h_cashflows[i] = p;
        }
    }

    double sum = 0.0;
    for (int i = 0; i < param.n_paths; ++i)
        sum += h_cashflows[i];

    return exp_min_r_dt * sum / (double) param.n_paths;
}


int main(int argc, char **argv) {

    double T = 1.00;
    double price = 0.0;

    Params param;
    param.n_timestamp = 100;
    param.n_paths = 32 * 1024;
    param.S0 = 3.60;
    param.strike_price = 4.00;
    param.dt = T / param.n_timestamp;
    param.R = 0.06;;
    param.sigma = 0.20;;
    param.pretty_print();
    {
        UnitCPUTime c;
        price = longstaff_schwartz_cpu(param);
        printf("CPU Longstaff-Schwartz: %.8lf\n", price);
    }

    return 0;
}
