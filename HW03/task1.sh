#!/usr/bin/env bash

#SBATCH -p wacc
#SBATCH -J Task1
#SBATCH -o Task1.out -e Task1.err
#SBATCH --gres=gpu:1

nvcc task1.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o task1
./task1 
