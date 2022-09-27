#!/usr/bin/env zsh

#SBATCH --job-name=Task3
#SBATCH --partition=wacc
#SBATCH --time=00-00:02:00
#SBATCH --gres=gpu:1
#SBATCH --output=Task3-%j.out
#SBATCH --error=Task3-%j.err

module load nvidia/cuda
nvcc task3.cu vscale.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o task3
./task3 1024
