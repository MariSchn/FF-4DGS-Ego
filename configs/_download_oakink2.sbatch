#!/bin/bash
#SBATCH --job-name=dl_oakink2
#SBATCH --account=3dv
#SBATCH --gpus=0
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=/home/dmonopoli/joblogs/download_oakink2_%j.out
#SBATCH --error=/home/dmonopoli/joblogs/download_oakink2_%j.err

set -uo pipefail

echo "=== DOWNLOADING OAKINK-V2 FROM HUGGINGFACE @ $(date) ==="
export HF_HOME="/home/dmonopoli/hf_cache"
export XDG_CACHE_HOME="/home/dmonopoli/xdg_cache"
mkdir -p "$HF_HOME" "$XDG_CACHE_HOME"

cd /home/dmonopoli/FF-4DGS-Ego
source /work/scratch/dmonopoli/venv_gb10/bin/activate

python scripts/download_oakink2.py

echo "=== OAKINK-V2 DOWNLOAD JOB DONE @ $(date) ==="
