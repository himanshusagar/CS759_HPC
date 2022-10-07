#!/usr/bin/env zsh

#SBATCH --job-name=Task2
#SBATCH --partition=wacc
#SBATCH --time=00-00:05:00
#SBATCH --gres=gpu:1
#SBATCH --output=Task2-%j.out
#SBATCH --error=Task2-%j.err


module load nvidia/cuda
nvcc task2.cu stencil.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std c++17 -o task2

./task2 8 2 6

# begin=$((2**10))
# end=$((2**30))

# for (( i=$begin ; i<=$end ; i=i*2 )); 
# do
#     ./task2 $i 128 1024
# done

