#!/usr/bin/env zsh

#SBATCH --job-name=Task1
#SBATCH --partition=wacc
#SBATCH --time=00-00:05:00
#SBATCH --output=Task1-%j.out
#SBATCH --error=Task1-%j.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1

rm task1

simd=true

if [ "$simd" = true ]
then
   g++ task1.cpp optimize.cpp -Wall -O3 -std=c++17 -o task1 -march=native -fopt-info-vec -ffast-math
else
   g++ task1.cpp optimize.cpp -Wall -O3 -std=c++17 -o task1 -fno-tree-vectorize
fi

./task1 1000000
