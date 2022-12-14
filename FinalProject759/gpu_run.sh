#!/usr/bin/env zsh

rm mc_gpu
nvcc mc_gpu.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std c++17 -o mc_gpu -lcurand 
./mc_gpu




