#!/usr/bin/env bash

#SBATCH -p wacc
#SBATCH -J Task1
#SBATCH -o Task1.out -e Task1.err

g++ scan.cpp task1.cpp -Wall -O3 -std=c++17 -o task1

begin=$((2**10))
end=$((2**30))

for (( i=$begin ; i<=$end ; i=i*2 )); 
do
    ./task1 $i
done