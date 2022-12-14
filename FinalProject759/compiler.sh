#!/usr/bin/env zsh

#SBATCH --job-name=Task
#SBATCH --partition=wacc
#SBATCH --time=00-00:05:00
#SBATCH --output=Task-%j.out
#SBATCH --error=Task-%j.err
#SBATCH --gres=gpu:1

git clone https://git.doit.wisc.edu/HSAGAR2/repo759
cd repo759
cd FinalProject759

mkdir -p external
cd external
wget https://github.com/Reference-LAPACK/lapack/archive/refs/tags/v3.11.tar.gz
tar -xvf v3.11.tar.gz
cd lapack-3.11/
cp make.inc.example make.inc  # use example make as make
make
cd ../../

mkdir -p external
cd external
wget http://www.netlib.org/blas/blas-3.11.0.tgz
tar -xvf blas-3.11.0.tgz  # unzip the blas source files
cd BLAS-3.11.0/ 
make
mv blas_LINUX.a libblas.a
cd ../../


rm mc_gpu
rm mc_cpu

module load nvidia/cuda/11.6.0  gcc/9.4.0 
nvcc mc_cpu.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std c++17 -o mc_cpu -llapack -lblas -lm -lcurand -lgfortran -Xcompiler -Lexternal/lapack-3.11  -Xcompiler -Lexternal/BLAS-3.11.0 
nvcc mc_gpu.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std c++17 -o mc_gpu -lcurand 
./mc_cpu
./mc_gpu
