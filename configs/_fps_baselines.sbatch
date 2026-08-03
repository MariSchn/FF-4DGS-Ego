#!/bin/bash
#SBATCH --job-name=fps_base
#SBATCH --account=3dv
#SBATCH --gpus=gb10:1
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/home/dmonopoli/joblogs/fps_baselines_%j.out
#SBATCH --error=/home/dmonopoli/joblogs/fps_baselines_%j.err

set -uo pipefail
cd /home/dmonopoli/FF-4DGS-Ego
source /work/scratch/dmonopoli/venv_gb10/bin/activate

echo "=== BENCHMARK STANDALONE BASELINE NN FPS @ $(date) ==="
python scripts/fps_probe_baselines.py
echo "=== BENCHMARK DONE @ $(date) ==="
