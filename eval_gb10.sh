#!/bin/bash
# ---------------------------------------------------------------------------
# Full-set hand-metric evaluation of a checkpoint on the gb10 (ARM) node.
# Runs scripts/eval_hand_head.py over the FULL locked val split (no 12-batch
# cap), reporting MPJPE / PA-MPJPE / MPVPE / AUC for left/right/all.
#
#   sbatch eval_gb10.sh configs/exp_p1a_abs3d.yaml checkpoints/p1a_abs3d/checkpoint_300.pt
# ---------------------------------------------------------------------------
#SBATCH --job-name=ff4dgs-eval
#SBATCH --account=3dv
#SBATCH --gpus=gb10:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail

CONFIG="${1:?usage: eval_gb10.sh <config.yaml> <ckpt.pt> [out.json]}"
CKPT="${2:?usage: eval_gb10.sh <config.yaml> <ckpt.pt> [out.json]}"
OUT="${3:-outputs/eval_$(basename "${CKPT%.pt}").json}"
mkdir -p logs outputs

echo "=================================================="
echo "Job    : ${SLURM_JOB_ID:-?} on $(hostname)"
echo "Config : ${CONFIG}"
echo "Ckpt   : ${CKPT}"
echo "Out    : ${OUT}"
echo "Started: $(date)"
echo "=================================================="

source venv_gb10/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

python3 -m scripts.eval_hand_head \
  --config "${CONFIG}" \
  --ckpt   "${CKPT}" \
  --batch-size 4 \
  --out    "${OUT}"

echo "=== RESULT (${OUT}) ==="
cat "${OUT}"
echo "Finished: $(date)"
