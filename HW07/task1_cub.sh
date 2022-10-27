#!/usr/bin/env zsh

#SBATCH --job-name=Task1
#SBATCH --partition=wacc
#SBATCH --time=00-00:05:00
#SBATCH --gres=gpu:1
#SBATCH --output=Task1-%j.out
#SBATCH --error=Task1-%j.err

module load nvidia/cuda/11.6.0  gcc/9.4.0 

rm task1_cub

nvcc task1_cub.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std c++17 -o task1_cub

#./task1_cub 1024

begin=$((2**10))
end=$((2**30))

for (( i=$begin ; i<=$end ; i=i*2 )); 
do
    ./task1_cub $i
done
