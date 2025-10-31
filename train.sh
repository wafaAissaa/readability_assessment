#!/bin/bash
#SBATCH --job-name=train-DL-0
#SBATCH --output="outputs/output_%j.txt"
#SBATCH --partition=gpu
#SBATCH --gres="gpu"
#SBATCH --qos="preemptible"
#SBATCH --mem-per-cpu=10000
#SBATCH --ntasks=1
#SBATCH --time=30:00:00
#SBATCH --constraint=TeslaA100|GeForceRTX3090|TeslaV100

# First argument passed to sbatch will be the OLL_alpha value
OLL_ALPHA=$1

# Print for clarity
echo "Running with OLL_alpha=${OLL_ALPHA}"

# Run your Python script
python main.py --OLL_alpha "${OLL_ALPHA}"
