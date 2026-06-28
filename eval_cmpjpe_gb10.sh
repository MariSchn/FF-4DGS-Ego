#!/bin/bash
# C/PA-MPJPE eval of the fine-tuned hand head on the held-out H2O subject4.
# Arg 1 = checkpoint (default: best_cmpjpe.pt from the cross-subject fine-tune).
#SBATCH --job-name=ff4dgs-cmpeval
#SBATCH --account=3dv
#SBATCH --gpus=gb10:1
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=/work/scratch/dmonopoli/joblogs/%j_cmpeval.out
#SBATCH --error=/work/scratch/dmonopoli/joblogs/%j_cmpeval.out
set -uo pipefail
cd /home/dmonopoli/FF-4DGS-Ego
source venv_gb10/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
CKPT="${1:-/work/scratch/dmonopoli/checkpoints/h2o_hand/best_cmpjpe.pt}"
echo "=== H2O C/PA-MPJPE eval (held-out subject4) $(date) ==="
echo "ckpt=$CKPT"
python3 -m scripts.eval_cmpjpe \
  --config configs/exp_h2o_hand.yaml \
  --ckpt "$CKPT" \
  --h2o /work/scratch/dmonopoli/h2o_packed \
  --subject subject4 \
  --limit 400
echo "=== done $(date) ==="
