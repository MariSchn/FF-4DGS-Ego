#!/bin/bash
# 21-joint (16 base + 5 fingertips) H2O eval on held-out subject4 — the airtight
# EgoGrasp comparison convention. Arg 1: "verify" (limit 8 + viz) or "full" (limit 400).
#SBATCH --job-name=ff4dgs-j21
#SBATCH --account=3dv
#SBATCH --gpus=gb10:1
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=/work/scratch/dmonopoli/joblogs/%j_j21.out
#SBATCH --error=/work/scratch/dmonopoli/joblogs/%j_j21.out
set -uo pipefail
cd /home/dmonopoli/FF-4DGS-Ego
source venv_gb10/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
CKPT=/work/scratch/dmonopoli/checkpoints/h2o_hand/best_cmpjpe.pt
MODE="${1:-verify}"
if [ "$MODE" = "verify" ]; then
  EXTRA="--viz /work/scratch/dmonopoli/joblogs/j21_viz.png --limit 8"
else
  EXTRA="--limit 400"
fi
echo "=== 21-joint eval ($MODE) $(date) ==="
python3 -m scripts.eval_cmpjpe \
  --config configs/exp_h2o_hand.yaml --ckpt "$CKPT" \
  --h2o /work/scratch/dmonopoli/h2o_packed --subject subject4 \
  --joints21 $EXTRA
echo "=== done $(date) ==="
