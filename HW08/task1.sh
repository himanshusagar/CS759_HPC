#!/usr/bin/env zsh

#SBATCH --job-name=Task1
#SBATCH --partition=wacc
#SBATCH --time=00-00:05:00
#SBATCH --cpus-per-task=20
#SBATCH --output=Task1-%j.out
#SBATCH --error=Task1-%j.err

rm task1
g++ task1.cpp matmul.cpp -Wall -O3 -std=c++17 -o task1 -fopenmp

begin=$((1))
end=$((20))

for (( i=$begin ; i<=$end ; i=i+1 )); 
do
    ./task1 1024 $i
done

