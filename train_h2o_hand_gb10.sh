#!/bin/bash
#SBATCH --job-name=ff4dgs-h2ohand
#SBATCH --account=3dv
#SBATCH --gpus=gb10:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/work/scratch/dmonopoli/joblogs/%j_h2ohand.out
#SBATCH --error=/work/scratch/dmonopoli/joblogs/%j_h2ohand.out
set -uo pipefail
cd /home/dmonopoli/FF-4DGS-Ego
source venv_gb10/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
echo "=== H2O hand fine-tune $(date) ==="
python3 -m scripts.train_h2o_hand --config configs/exp_h2o_hand.yaml
echo "=== done $(date) ==="
