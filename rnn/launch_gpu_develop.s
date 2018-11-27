#!/bin/bash

#SBATCH --job-name=lab1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10GB
#SBATCH --gres=gpu
#SBATCH --time=01:00:00
#SBATCH --output=out.%j


module purge
module load cuda/9.0.176  cudnn/9.0v7.0.5
module load pytorch/python3.6/0.3.0_4
#python train.py
python test.py
