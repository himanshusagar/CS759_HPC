#!/usr/bin/env zsh

#SBATCH --job-name=Task3
#SBATCH --partition=wacc
#SBATCH --time=00-00:05:00
#SBATCH --output=Task3-%j.out
#SBATCH --error=Task3-%j.err
#SBATCH --ntasks-per-node=2

module load mpi/mpich/4.0.2

rm task3
mpicxx task3.cpp -Wall -O3 -o task3

begin=$((2**1))
end=$((2**26))

for (( i=$begin ; i<=$end ; i=i*2 )); 
do
    srun -n 2 task3 $i
done

