#!/bin/bash
# ---------------------------------------------------------------------------
# P3 REAL: partial unfreeze (last 4 blocks) + GS head + GT object-depth
# supervision (Cyrus direction a). The main-track metric-scene shot. Logs to
# /work/scratch ($HOME inode quota makes $HOME/logs silently fail <1s).
#   sbatch train_p3_gtdepth_gb10.sh
# After it finishes (or checkpoints), run B2 (eval_scene_metric_gt) on
# checkpoints/p3_gtdepth/best_mpjpe.pt: does object depth drop below 62cm?
# ---------------------------------------------------------------------------
#SBATCH --job-name=ff4dgs-p3gtd
#SBATCH --account=3dv
#SBATCH --gpus=gb10:1
#SBATCH --mem=32G
#SBATCH --time=23:59:00

set -euo pipefail
cd /home/dmonopoli/FF-4DGS-Ego
mkdir -p /work/scratch/dmonopoli/joblogs
exec > >(tee /work/scratch/dmonopoli/joblogs/${SLURM_JOB_ID}_p3gtd.out) 2>&1

source venv_gb10/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1

echo "=== P3 GTDEPTH start $(date) ==="
python3 -m scripts.train_hand_head --config configs/exp_p3_gtdepth_unfreeze.yaml
echo "=== P3 GTDEPTH done $(date) ==="
