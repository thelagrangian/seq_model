#!/bin/bash

#SBATCH --job-name=lab2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10GB
#SBATCH --gres=gpu:p40:1
#SBATCH --time=05:00:00
#SBATCH --output=out.%j

module purge
module load cuda/9.0.176  cudnn/9.0v7.0.5

python char_cnn_test.py
# --dataset ptb
