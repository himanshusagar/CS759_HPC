#!/usr/bin/env zsh

#SBATCH --job-name=Task1
#SBATCH --partition=wacc
#SBATCH --time=00-00:02:00
#SBATCH --gres=gpu:1
#SBATCH --output=Task1-%j.out
#SBATCH --error=Task1-%j.err


module load nvidia/cuda
nvcc task1.cu matmul.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std c++17 -o task1

./task1 
