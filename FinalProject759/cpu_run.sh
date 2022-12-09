#!/usr/bin/env zsh

#SBATCH --job-name=Task
#SBATCH --partition=wacc
#SBATCH --time=00-00:05:00
#SBATCH --output=Task-%j.out
#SBATCH --error=Task-%j.err
#SBATCH --gres=gpu:1

module load nvidia/cuda/11.6.0  gcc/9.4.0 
rm mc_cpu
rm mc_gpu

module load nvidia/cuda
nvcc mc_cpu.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std c++17 -o mc_cpu -llapack -lblas -lm -lcurand -lgfortran -Xcompiler -Lexternal/lapack-3.11.0  -Xcompiler -Lexternal/BLAS-3.11.0
./mc_cpu

