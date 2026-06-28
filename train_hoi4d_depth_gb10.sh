#!/bin/bash
# HOI4D dense-depth experiment: train gs_head + last-4 blocks on HOI4D sensor depth.
#SBATCH --job-name=ff4dgs-hoi4ddep
#SBATCH --account=3dv
#SBATCH --gpus=gb10:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/work/scratch/dmonopoli/joblogs/%j_hoi4ddepth.out
#SBATCH --error=/work/scratch/dmonopoli/joblogs/%j_hoi4ddepth.out
set -euo pipefail
cd /home/dmonopoli/FF-4DGS-Ego
source venv_gb10/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
echo "=== HOI4D dense-depth train $(date) ==="
python3 -m scripts.train_hoi4d_depth --config configs/exp_hoi4d_depth.yaml
echo "=== done $(date) ==="
