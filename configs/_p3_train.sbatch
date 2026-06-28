#!/bin/bash
#SBATCH --account=3dv
#SBATCH --partition=jobs
#SBATCH --gpus=gb10:1
# ---------------------------------------------------------------------------
# Generic P3 train wrapper. Usage:
#   sbatch --job-name=X --output=/home/dmonopoli/X_%j.out --error=... --time=HH:MM:SS \
#          _p3_train.sbatch <config.yaml> [key=val overrides ...]
# Runs scripts.train_hand_head on a gb10 node (aarch64 venv). Checkpoints should
# be redirected to node-local $TMPDIR via a training.output_dir=... override on
# full runs (per-user /work quota can silently cancel jobs); the metrics we want
# (val C-MPJPE, obj_depth_residual_m, scale_residual_m) land in this .out log.
# ---------------------------------------------------------------------------
set -uo pipefail
CFG="${1:?config required}"
shift || true
echo "=== P3 JOB $SLURM_JOB_ID on $(hostname) @ $(date) ==="
echo "=== CONFIG: $CFG  OVERRIDES: $* ==="
cd /home/dmonopoli/FF-4DGS-Ego || exit 2
source venv_gb10/bin/activate
echo "PYTHON: $(which python)  ARCH: $(uname -m)"
python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),'dev',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
OUT="${TMPDIR:-/tmp}/p3ckpt_${SLURM_JOB_ID}"
mkdir -p "$OUT"
echo "=== node-local output_dir: $OUT ===" ; df -h "$OUT" | tail -1
echo "=== TRAIN START @ $(date) ==="
# Force checkpoints/vis to node-local $TMPDIR ($HOME is inode-exhausted, /work
# per-user quota can silently cancel). Caller overrides still win (appended last).
python -m scripts.train_hand_head --config "$CFG" training.output_dir="$OUT" "$@"
RC=$?
echo "=== P3 JOB $SLURM_JOB_ID rc=$RC COMPLETE @ $(date) ==="
exit $RC
