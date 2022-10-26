#!/usr/bin/env zsh

#SBATCH --job-name=Task3
#SBATCH --partition=wacc
#SBATCH --time=00-00:05:00
#SBATCH --nodes=1 --cpus-per-task=4
#SBATCH --output=Task3-%j.out
#SBATCH --error=Task3-%j.err

rm task3

g++ task3.cpp -Wall -O3 -std=c++17 -o task3 -fopenmp

./task3 10

