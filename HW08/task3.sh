#!/usr/bin/env zsh

#SBATCH --job-name=Task3
#SBATCH --partition=wacc
#SBATCH --time=00-00:05:00
#SBATCH --cpus-per-task=20
#SBATCH --output=Task3-%j.out
#SBATCH --error=Task3-%j.err

rm task3
g++ task3.cpp msort.cpp -Wall -O3 -std=c++17 -o task3 -fopenmp

./task3 12 3 1


