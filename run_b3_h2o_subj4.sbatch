#!/bin/bash
#SBATCH --account=3dv
#SBATCH --gpus=gb10:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --job-name=b3h2o
#SBATCH --output=/work/scratch/dmonopoli/joblogs/b3_%j.out
#SBATCH --error=/work/scratch/dmonopoli/joblogs/b3_%j.err
# ---------------------------------------------------------------------------
# B3: the REAL held-out H2O absolutes for the scale table (Table 2).
#
# The 58.0 / 36.6 (C-MPJPE / WRIST) in the table were measured on the MIXED
# all-subjects eval set, not the clean held-out subject-4 split. This evals the
# H2O fine-tuned head on subject-4 only (--subject subject4, 21-joint) to get the
# true held-out PA / C-MPJPE / WRIST, plus the warm-start (default) head on the same
# clips as the baseline (expected ~233/291, per hires-train-result.md).
#
# CAVEAT to read with the result: this is a valid held-out number ONLY if
# best_cmpjpe.pt was trained WITHOUT subject-4. If its C-MPJPE comes back close to
# the mixed-set 58, train/test likely overlapped -> a subject1-only retrain is then
# needed for a clean number. The default-head row (never saw H2O) is always held-out.
#
# Submit from the login node:  sbatch run_b3_h2o_subj4.sbatch
# ---------------------------------------------------------------------------
set -uo pipefail
source /work/scratch/dmonopoli/venv_gb10/bin/activate
export XDG_CACHE_HOME=/tmp/xc TORCH_HOME=/tmp/th HF_HOME=/tmp/hf MPLCONFIGDIR=/tmp/mpl PYTHONUNBUFFERED=1
mkdir -p /tmp/xc /tmp/th /tmp/hf /tmp/mpl
cd /home/dmonopoli/FF-4DGS-Ego
echo "### B3 H2O subject-4 held-out eval | $(hostname) | $(date) | commit $(git rev-parse --short HEAD) ###"

H2O=/work/scratch/dmonopoli/h2o_packed
CFG=configs/exp_h2o_hand.yaml
FT=/work/scratch/dmonopoli/checkpoints/h2o_hand/best_cmpjpe.pt
DEF=/work/courses/3dv/team25/checkpoints/default/hand_head_final.pt

echo "==== (1) FINE-TUNED head (best_cmpjpe.pt) on subject-4, 21-joint ===="
[ -f "$FT" ] && python -u -m scripts.eval_cmpjpe --config "$CFG" --ckpt "$FT" \
  --h2o "$H2O" --subject subject4 --joints21 \
  || echo "[b3] fine-tuned head missing or eval failed: $FT"

echo ""
echo "==== (2) DEFAULT warm-start head (never saw H2O) on subject-4, 21-joint -- baseline ===="
python -u -m scripts.eval_cmpjpe --config "$CFG" --ckpt "$DEF" \
  --h2o "$H2O" --subject subject4 --joints21 \
  || echo "[b3] baseline eval failed"

echo ""
echo "### DONE @ $(date) ###"
echo ">>> Read C-MPJPE / WRIST / PA-MPJPE for the FINE-TUNED head: these are the true"
echo ">>> held-out Table-2 numbers (replace the mixed-set 58.0 / 36.6). If fine-tuned"
echo ">>> C-MPJPE ~= 58, suspect train/test overlap -> needs a subject1-only retrain."
