#!/bin/bash
#SBATCH --job-name=hawor
#SBATCH --partition=jobs
#SBATCH --account=3dv
#SBATCH --gpus=2080ti:1
#SBATCH --mem=40G
#SBATCH --time=08:00:00
# NOTE: /work is at disk quota (SLURM can't create logs there -> instant FAILED 0:53).
# Write the single job log to HOME (1 inode, fine).
#SBATCH --output=/home/dmonopoli/HaWoR/hawor_eval/%j_hawor.out
#SBATCH --error=/home/dmonopoli/HaWoR/hawor_eval/%j_hawor.out
# NB: no `set -u` — sourced venv/pip/conda-free activate scripts reference unset vars.
set -o pipefail

HAWOR=/home/dmonopoli/HaWoR
# /work is at user disk quota -> persist only small scripts+outputs in HOME (inode-cheap:
# a handful of files), all heavy intermediates (venv, frames, SLAM) on node-local TMPDIR.
WORK=/home/dmonopoli/HaWoR/hawor_eval
mkdir -p "$WORK"
VENV=$TMPDIR/hvenv
export HF_HOME=$TMPDIR/hf
export TORCH_HOME=$TMPDIR/torch
mkdir -p "$HF_HOME" "$TORCH_HOME"

echo "############ NODE $(hostname) $(date) ############"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

# ---------- 1. BUILD ENV (DROID-SLAM + lietorch CUDA build) ----------
source "$WORK/build_hawor_env.sh" "$VENV" || { echo "BUILD FAILED"; exit 11; }

# CUDA_HOME / PATH / LD_LIBRARY_PATH were exported by the sourced build script.
cd "$HAWOR"

# ---------- MILESTONE 1: bundled example video ----------
# Copy the example mp4 to TMPDIR so HaWoR writes extracted_images/SLAM/tracks node-local
# (avoids spamming HOME inodes with thousands of jpgs).
echo "############ MILESTONE 1: example/video_0.mp4 ############"
EXDIR=$TMPDIR/example; mkdir -p "$EXDIR"
cp "$HAWOR/example/video_0.mp4" "$EXDIR/"
python "$WORK/run_hawor_infer.py" \
    --video_path "$EXDIR/video_0.mp4" \
    --out "$WORK/example_video_0_pred.npz" 2>&1 | tail -40
echo "M1 exit=$?"
ls -la "$WORK/example_video_0_pred.npz" 2>&1

# ---------- MILESTONE 2: H2O subject4 clips ----------
echo "############ MILESTONE 2: H2O subject4 clips ############"
CLIPDIR=$TMPDIR/h2o_clips
python "$WORK/extract_h2o_clips_mp4.py" \
    --h2o /work/scratch/dmonopoli/h2o_packed --subject subject4 \
    --out "$CLIPDIR" --n-clips 6 --max-frames 250 2>&1 | tail -20

# copy GT npz to persistent WORK for the metric step
cp "$CLIPDIR"/*_gt.npz "$WORK/" 2>/dev/null || true

for mp4 in "$CLIPDIR"/*.mp4; do
    [ -e "$mp4" ] || continue
    clip=$(basename "$mp4" .mp4)
    echo "==== HaWoR on $clip ===="
    # H2O packed crop is 224x224 from a 720-tall center crop. Pass true focal so SLAM/
    # motion-est use metric intrinsics (else HaWoR guesses focal=600). fx_adj from GT npz.
    FOCAL=$(python - "$WORK/${clip}_gt.npz" <<'PY'
import sys, numpy as np
print(f"{float(np.load(sys.argv[1])['Kadj'][0]):.4f}")
PY
)
    echo "  using img_focal=$FOCAL"
    timeout 2400 python "$WORK/run_hawor_infer.py" \
        --video_path "$mp4" --img_focal "$FOCAL" \
        --out "$WORK/${clip}_pred.npz" 2>&1 | tail -25
    echo "  $clip exit=$?"
done

echo "############ DONE $(date) ############"
ls -la "$WORK"/*_pred.npz 2>&1
