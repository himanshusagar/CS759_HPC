#!/usr/bin/env zsh

#SBATCH --job-name=Task1
#SBATCH --partition=wacc
#SBATCH --time=00-00:05:00
#SBATCH --gres=gpu:1
#SBATCH --output=Task1-%j.out
#SBATCH --error=Task1-%j.err

module load nvidia/cuda
rm task1
nvcc task1.cu mmul.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -lcublas -std c++17 -o task1

begin=$((2**5))
end=$((2**15))

for (( i=$begin ; i<=$end ; i=i*2 )); 
do
    ./task1 $i 5
done


