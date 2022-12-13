#!/usr/bin/env zsh

#SBATCH --job-name=TaskGPU
#SBATCH --partition=wacc
#SBATCH --time=00-00:05:00
#SBATCH --output=TaskGPU-%j.out
#SBATCH --error=TaskGPU-%j.err
#SBATCH --gres=gpu:1

module load nvidia/cuda/11.6.0  gcc/9.4.0 
rm mc_gpu

module load nvidia/cuda
nvcc mc_gpu.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std c++17 -o mc_gpu -lm -lcurand 
./mc_gpu

