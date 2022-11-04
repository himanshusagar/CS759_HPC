#!/usr/bin/env zsh

#SBATCH --job-name=Task3
#SBATCH --partition=wacc
#SBATCH --time=00-00:05:00
#SBATCH --cpus-per-task=20
#SBATCH --output=Task3-%j.out
#SBATCH --error=Task3-%j.err

rm task3
g++ task3.cpp msort.cpp -Wall -O3 -std=c++17 -o task3 -fopenmp

begin=$((2**1))
end=$((2**21))

for (( i=$begin ; i<=$end ; i=i*2 )); 
do
    ./task3 1000000 8 $i
done

