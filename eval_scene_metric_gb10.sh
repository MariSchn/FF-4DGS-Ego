#!/bin/bash
# ---------------------------------------------------------------------------
# SCENE-METRIC experiment (the headline test of the metric-coupling thesis):
# does the hand-depth anchor make the predicted GS SCENE metric, not just the
# hand? Runs scripts/eval_metric_scale.py on TWO checkpoints back-to-back:
#
#   BASELINE (no anchor)  = Marian's 50mm hand_head_final.pt (trained w/o HDGLA)
#   WITH ANCHOR           = our P2 warm-start best_mpjpe.pt   (HDGLA, scene_follows_hand)
#
# Read-out per run:
#   * global scale s = median(hand_z / gs_depth_at_hand): with the anchor the
#     scene should ALREADY be metric -> s ~ 1; without it -> s far from 1.
#   * hand-depth residual PRE (raw gs_depth vs metric hand) vs POST (after s):
#     a small PRE residual means the scene depth is already at metric scale.
# A large gap (his big PRE residual / s!=1  vs  our small PRE residual / s~1)
# is the evidence that the coupling makes the SCENE metric.
#
#   sbatch eval_scene_metric_gb10.sh
# ---------------------------------------------------------------------------
#SBATCH --job-name=ff4dgs-scenemetric
#SBATCH --account=3dv
#SBATCH --gpus=gb10:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail

CFG="${1:-configs/exp_p2_pinhole_abs3d.yaml}"
BASELINE="${2:-/work/courses/3dv/team25/checkpoints/default/hand_head_final.pt}"
ANCHORED="${3:-checkpoints/p2_warmstart/best_mpjpe.pt}"
NCLIPS="${4:-100}"
mkdir -p logs

echo "=================================================="
echo "Job     : ${SLURM_JOB_ID:-?} on $(hostname)"
echo "Config  : ${CFG} | num_clips=${NCLIPS}"
echo "Started : $(date)"
echo "=================================================="

source venv_gb10/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

echo ""
echo "########## BASELINE — NO anchor (Marian 50mm: ${BASELINE}) ##########"
python3 -m scripts.eval_metric_scale --config "${CFG}" --checkpoint "${BASELINE}" --num_clips "${NCLIPS}" --max_seqs 30

echo ""
echo "########## WITH ANCHOR — P2 warm-start (${ANCHORED}) ##########"
python3 -m scripts.eval_metric_scale --config "${CFG}" --checkpoint "${ANCHORED}" --num_clips "${NCLIPS}" --max_seqs 30

echo "Finished: $(date)"
