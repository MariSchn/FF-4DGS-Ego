#!/bin/bash
# ---------------------------------------------------------------------------
# B1 stage 1 (gb10 / venv_gb10): dump per-clip frames + hand anchor targets
# into a portable bundle the x86 UniDepth eval reads. See publication-plan E1.
#
#   sbatch extract_b1_gb10.sh [cfg] [ckpt] [num_clips] [out]
# ---------------------------------------------------------------------------
#SBATCH --job-name=ff4dgs-b1extract
#SBATCH --account=3dv
#SBATCH --gpus=gb10:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail

CFG="${1:-configs/exp_p2_pinhole_warmstart.yaml}"
CKPT="${2:-checkpoints/p2_warmstart/best_mpjpe.pt}"
NCLIPS="${3:-60}"
OUT="${4:-/work/scratch/dmonopoli/b1_bundle.pt}"   # scratch: bundle is ~1.4G, keep it off the $HOME quota
mkdir -p logs outputs

echo "=================================================="
echo "Job     : ${SLURM_JOB_ID:-?} on $(hostname)"
echo "Config  : ${CFG} | ckpt=${CKPT} | num_clips=${NCLIPS} | out=${OUT}"
echo "Started : $(date)"
echo "=================================================="

source venv_gb10/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

python3 -m scripts.extract_b1_frames \
    --config "${CFG}" --checkpoint "${CKPT}" \
    --num_clips "${NCLIPS}" --max_seqs 30 --out "${OUT}"

echo "Finished: $(date)"
