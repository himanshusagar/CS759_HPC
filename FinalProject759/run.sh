#!/usr/bin/env zsh

#SBATCH --job-name=Task
#SBATCH --partition=wacc
#SBATCH --time=00-00:05:00
#SBATCH --output=Task-%j.out
#SBATCH --error=Task-%j.err
#SBATCH --gres=gpu:1

module load nvidia/cuda/11.6.0  gcc/9.4.0 
rm run.out


