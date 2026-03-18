#!/bin/bash
#SBATCH -A 3dv
#SBATCH --job-name=hand_train_T25
#SBATCH --partition=jobs
#SBATCH --gpus=5060ti:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

# Carichiamo l'ambiente
source ~/miniconda3/bin/activate neoverse

# --- QUESTA È LA RIGA MAGICA ---
export PYTHONPATH=$PYTHONPATH:$(pwd)
# -------------------------------

cd ~/FF-4DGS-Ego
mkdir -p logs

export PYTORCH_ALLOC_CONF=expandable_segments:True

python scripts/train_hand_head.py \
    --seq_path ~/FF-4DGS-Ego/data/hot3d_aria/sequences/P0001_10a27bf7/ \
    -- num_frames 4 \
    --checkpoint models/reconstructor.ckpt