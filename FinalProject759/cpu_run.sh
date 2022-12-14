#!/usr/bin/env zsh

rm mc_gpu
nvcc mc_cpu.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std c++17 -o mc_cpu -llapack -lblas -lm -lcurand -lgfortran -Xcompiler -Lexternal/lapack-3.11  -Xcompiler -Lexternal/BLAS-3.11.0 
./mc_cpu
