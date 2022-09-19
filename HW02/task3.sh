#!/usr/bin/env bash

#SBATCH -p wacc
#SBATCH -J Task3
#SBATCH -o Task3.out -e Task3.err

g++ task3.cpp matmul.cpp -Wall -O3 -std=c++17 -o task3

./task3
