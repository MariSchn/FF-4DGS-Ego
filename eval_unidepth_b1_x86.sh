#!/bin/bash
# ---------------------------------------------------------------------------
# B1 stage 2 (x86 jobs node / venv_unidepth): run UniDepthV2 on the extracted
# bundle and compare its metric depth AT the hand to our anchor's. See E1.
# NOTE: x86 node (UniDepth env is x86) — plain --gpus=1 routes to the jobs
# partition; do NOT request gb10 here.
#
#   sbatch eval_unidepth_b1_x86.sh [bundle] [weights] [out]
# ---------------------------------------------------------------------------
#SBATCH --job-name=ff4dgs-b1unidepth
#SBATCH --account=3dv
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail

BUNDLE="${1:-/work/scratch/dmonopoli/b1_bundle.pt}"
WEIGHTS="${2:-lpiccinelli/unidepth-v2-vitl14}"
OUT="${3:-outputs/b1_unidepth.json}"
mkdir -p logs outputs

echo "=================================================="
echo "Job     : ${SLURM_JOB_ID:-?} on $(hostname) [$(uname -m)]"
echo "Bundle  : ${BUNDLE} | weights=${WEIGHTS} | out=${OUT}"
echo "Started : $(date)"
echo "=================================================="

source "$HOME/venv_unidepth/bin/activate"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
# Use cached HF weights (pre-downloaded on the login node) — no node internet needed.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

cd "$HOME/FF-4DGS-Ego"
PYTHONPATH="$HOME/UniDepth:${PYTHONPATH:-}" python3 scripts/eval_unidepth_b1.py \
    --bundle "${BUNDLE}" --weights "${WEIGHTS}" --out "${OUT}"

echo "Finished: $(date)"
