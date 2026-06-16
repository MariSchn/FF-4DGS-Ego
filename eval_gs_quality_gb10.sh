#!/bin/bash
# ---------------------------------------------------------------------------
# E3 scene-quality: did adding the metric anchor HURT GS rendering quality?
# Runs scripts/eval_gs_head.py (full-frame PSNR/SSIM/LPIPS from the re-rendered
# splats) on TWO checkpoints back-to-back:
#
#   BASELINE (no anchor) = Marian's hand_head_final.pt (untrained hand_to_gs)
#   WITH ANCHOR          = our p2_warmstart best_mpjpe.pt (anchor-trained injection)
#
# E3 target: PSNR/SSIM within ~noise (<=~0.5dB) of baseline => the metric anchor
# does not trade away scene quality. NOTE: backbone is frozen; only the
# hand_to_gs injection differs, so any delta isolates the injection's effect on
# the rendered scene.
#
#   sbatch eval_gs_quality_gb10.sh [cfg] [baseline_ckpt] [anchored_ckpt] [limit_clips]
# ---------------------------------------------------------------------------
#SBATCH --job-name=ff4dgs-gsquality
#SBATCH --account=3dv
#SBATCH --gpus=gb10:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail

CFG="${1:-configs/exp_p2_pinhole_warmstart.yaml}"
BASELINE="${2:-/work/courses/3dv/team25/checkpoints/default/hand_head_final.pt}"
ANCHORED="${3:-checkpoints/p2_warmstart/best_mpjpe.pt}"
LIMIT="${4:-400}"
mkdir -p logs outputs

echo "=================================================="
echo "Job     : ${SLURM_JOB_ID:-?} on $(hostname)"
echo "Config  : ${CFG} | limit_clips=${LIMIT}"
echo "Started : $(date)"
echo "=================================================="

source venv_gb10/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

echo ""
echo "########## BASELINE — NO anchor (${BASELINE}) ##########"
python3 -m scripts.eval_gs_head --config "${CFG}" --ckpt "${BASELINE}" \
    --limit-clips "${LIMIT}" --out outputs/e3_gs_baseline.json

echo ""
echo "########## WITH ANCHOR — p2_warmstart (${ANCHORED}) ##########"
python3 -m scripts.eval_gs_head --config "${CFG}" --ckpt "${ANCHORED}" \
    --limit-clips "${LIMIT}" --out outputs/e3_gs_anchored.json

echo "Finished: $(date)"
