#!/bin/bash

# Select which GPU to use
#SBATCH --gpus=5060ti:1
# SBATCH --gpus=gb10:1

#SBATCH --time=24:00:00
#SBATCH --account=3dv
#SBATCH --job-name=ff4dgs
#SBATCH --output=logs/%j.out    # Make sure the logs directory exists!
#SBATCH --error=logs/%j.err

source venv/bin/activate
python3 -m scripts.train_hand_head --config configs/train_hand_head.yaml
