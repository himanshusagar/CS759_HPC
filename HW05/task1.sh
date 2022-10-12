#!/usr/bin/env zsh

#SBATCH --job-name=Task1
#SBATCH --partition=wacc
#SBATCH --time=00-00:05:00
#SBATCH --gres=gpu:1
#SBATCH --output=Task1-%j.out
#SBATCH --error=Task1-%j.err

module load nvidia/cuda
nvcc task1.cu reduce.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std c++17 -o task1

begin=$((2**10))
end=$((2**30))

for (( i=$begin ; i<=$end ; i=i*2 )); 
do
    ./task1 $i 512
done
