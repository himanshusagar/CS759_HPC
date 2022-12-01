#!/usr/bin/env zsh

#SBATCH --job-name=Task2
#SBATCH --partition=wacc
#SBATCH --time=00-00:05:00
#SBATCH --output=Task2-%j.out
#SBATCH --error=Task2-%j.err
#SBATCH --nodes=2 --cpus-per-task=20 --ntasks-per-node=1

module load mpi/mpich/4.0.2

rm task2
mpicxx task2.cpp reduce.cpp -Wall -O3 -o task2 -fopenmp -fno-tree-vectorize -march=native -fopt-info-vec

# for i in {1..20}
# do
# 	srun -n 2 --cpu-bind=none ./task2 10000000 $i
# done

begin=$((2**1))
end=$((2**26))

for (( i=$begin ; i<=$end ; i=i*2 )); 
do
   srun -n 2 --cpu-bind=none ./task2 $i 9
done
