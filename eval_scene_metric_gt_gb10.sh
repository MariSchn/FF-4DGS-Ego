#!/bin/bash
# ---------------------------------------------------------------------------
# B2 / E2: NON-circular scene-metric on OBJECTS (publication-plan E2).
# Renders GT object depth (HOT3D meshes + per-frame poses) and compares each
# model's gs_depth to it in object regions (HAND excluded). Runs on TWO ckpts:
#
#   BASELINE (no anchor) = Marian's hand_head_final.pt
#   WITH ANCHOR          = our p2_warmstart best_mpjpe.pt
#
# E2 target: anchored object-region depth error CLEARLY below no-anchor
# (e.g. anchored ~8-12cm vs no-anchor ~20-30cm; delta<10cm up >=15pts) =>
# the hand anchor made the SCENE metric, not just the hand.
#
#   sbatch eval_scene_metric_gt_gb10.sh [cfg] [baseline] [anchored] [num_clips]
# ---------------------------------------------------------------------------
#SBATCH --job-name=ff4dgs-scenemetricgt
#SBATCH --account=3dv
#SBATCH --gpus=gb10:1
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail

CFG="${1:-configs/exp_p2_pinhole_warmstart.yaml}"
BASELINE="${2:-/work/courses/3dv/team25/checkpoints/default/hand_head_final.pt}"
ANCHORED="${3:-checkpoints/p2_warmstart/best_mpjpe.pt}"
NCLIPS="${4:-80}"
OBJ="${OBJ:-/work/scratch/dmonopoli/hot3d_assets}"
mkdir -p logs outputs

echo "=================================================="
echo "Job     : ${SLURM_JOB_ID:-?} on $(hostname)"
echo "Config  : ${CFG} | num_clips=${NCLIPS} | objects=${OBJ}"
echo "Started : $(date)"
echo "=================================================="

source venv_gb10/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

echo ""
echo "########## BASELINE — NO anchor (${BASELINE}) ##########"
python3 -m scripts.eval_scene_metric_gt --config "${CFG}" --checkpoint "${BASELINE}" \
    --objects_dir "${OBJ}" --num_clips "${NCLIPS}" --max_seqs 30 \
    --out outputs/e2_scene_metric_baseline.json

echo ""
echo "########## WITH ANCHOR — p2_warmstart (${ANCHORED}) ##########"
python3 -m scripts.eval_scene_metric_gt --config "${CFG}" --checkpoint "${ANCHORED}" \
    --objects_dir "${OBJ}" --num_clips "${NCLIPS}" --max_seqs 30 \
    --out outputs/e2_scene_metric_anchored.json

echo "Finished: $(date)"
