#!/bin/bash
# ---------------------------------------------------------------------------
# SCALEHEAD SMOKE: verify the scale-head pipeline end-to-end on gb10
# (3 seqs, one short epoch) BEFORE the 7h real run. Logs to /work/scratch
# (the $HOME inode quota makes $HOME/logs silently fail <1s).
#   sbatch train_p3_smoke_gb10.sh
# PASS criteria in the log: "Partial unfreeze: last 4 frame+global blocks",
# a non-zero obj_depth term in the accum print, finite grad norms, no NaN.
# ---------------------------------------------------------------------------
#SBATCH --job-name=ff4dgs-shsmoke
#SBATCH --account=3dv
#SBATCH --gpus=gb10:1
#SBATCH --mem=32G
#SBATCH --time=00:40:00

set -euo pipefail
cd /home/dmonopoli/FF-4DGS-Ego
mkdir -p /work/scratch/dmonopoli/joblogs
exec > >(tee /work/scratch/dmonopoli/joblogs/${SLURM_JOB_ID}_shsmoke.out) 2>&1

source venv_gb10/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1

echo "=== SCALEHEAD SMOKE start $(date) ==="
python3 -m scripts.train_hand_head --config configs/exp_p3_scalehead_smoke.yaml
echo "=== SCALEHEAD SMOKE done $(date) ==="
