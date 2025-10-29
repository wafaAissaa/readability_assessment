#!/bin/bash
#SBATCH --job-name=train-DL-0
#SBATCH --output="outputs/output0.txt"
#SBATCH --partition=gpu
#SBATCH --gres="gpu"
#SBATCH --qos="preemptible"
#SBATCH --mem-per-cpu=10000
#SBATCH --ntasks=1
#SBATCH --constraint=TeslaA100|GeForceRTX3090|TeslaV100

python train.py