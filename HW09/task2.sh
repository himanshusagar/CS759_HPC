#!/usr/bin/env zsh

#SBATCH --job-name=Task2
#SBATCH --partition=wacc
#SBATCH --time=00-00:05:00
#SBATCH --cpus-per-task=10
#SBATCH --output=Task2-%j.out
#SBATCH --error=Task2-%j.err

rm task2
g++ task2.cpp montecarlo.cpp -Wall -O3 -std=c++17 -o task2 -fopenmp -fno-tree-vectorize -march=native -fopt-info-vec

for i in {1..10}
do
	./task2 1000000 $i
done
