#!/usr/bin/env zsh

#SBATCH --job-name=Task2
#SBATCH --partition=wacc
#SBATCH --time=00-00:05:00
#SBATCH --cpus-per-task=20
#SBATCH --output=Task2-%j.out
#SBATCH --error=Task2-%j.err

rm task2
g++ task2.cpp convolution.cpp -Wall -O3 -std=c++17 -o task2 -fopenmp

begin=$((1))
end=$((20))

for (( i=$begin ; i<=$end ; i=i+1 )); 
do
    ./task2 1024 $i
done

