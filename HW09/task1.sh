#!/usr/bin/env zsh

#SBATCH --job-name=Task1
#SBATCH --partition=wacc
#SBATCH --time=00-00:05:00
#SBATCH --cpus-per-task=10
#SBATCH --output=Task1-%j.out
#SBATCH --error=Task1-%j.err

rm task1
g++ task1.cpp cluster.cpp -Wall -O3 -std=c++17 -o task1 -fopenmp

for i in {1..10}
do
	./task1 5040000 $i
done

