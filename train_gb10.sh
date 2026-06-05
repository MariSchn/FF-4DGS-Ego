#!/bin/bash
# ---------------------------------------------------------------------------
# Train the hand-scene metric-coupling model (L1 anchor + GS head) on a GB10
# node of the ETH student cluster.
#
#   sbatch train_gb10.sh                                   # ablation config
#   sbatch train_gb10.sh configs/my_other_config.yaml      # custom config
#
# Prereqs (one-time, on a login node):
#   ./prep_venv.sh                 # builds venv_gb10 (torch 2.7.1 cu128)
#   wandb login                    # for W&B logging (config: wandb.enabled)
#   models/NeoVerse/reconstructor.ckpt and models/MANO/ present
#   data.data_root in the config points at your sequences
#
# gb10 is available on both the 'interactive' and the default 'jobs' partition
# (studgpu-spark[01-06]); omitting --partition uses 'jobs', matching batch.sh.
# ---------------------------------------------------------------------------
#SBATCH --job-name=ff4dgs-gb10
#SBATCH --account=3dv
#SBATCH --gpus=gb10:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail

CONFIG="${1:-configs/ablation_hand_to_gs_injection.yaml}"
mkdir -p logs

echo "=================================================="
echo "Job        : ${SLURM_JOB_ID:-?} on $(hostname)"
echo "GPU(s)     : ${CUDA_VISIBLE_DEVICES:-?}"
echo "Config     : ${CONFIG}"
echo "Git        : $(git rev-parse --short HEAD 2>/dev/null || echo n/a) ($(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo n/a))"
echo "Started    : $(date)"
echo "=================================================="
nvidia-smi || true

# --- Environment (GB10-specific venv from prep_venv.sh) -------------------
if [[ ! -f venv_gb10/bin/activate ]]; then
  echo "ERROR: venv_gb10 not found. Run ./prep_venv.sh first." >&2
  exit 1
fi
source venv_gb10/bin/activate

# GB10 has ~128 GB unified (Grace+Blackwell) memory; reduce allocator
# fragmentation so large batches survive transient peaks.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TORCH_SHOW_CPP_STACKTRACES=1

# --- Fail fast, before the GPU slot is wasted -----------------------------
python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA unavailable in venv_gb10 — re-check prep_venv.sh"
print("torch", torch.__version__,
      "| device", torch.cuda.get_device_name(0),
      "| capability", torch.cuda.get_device_capability(0))
PY

test -f models/NeoVerse/reconstructor.ckpt \
  || { echo "ERROR: missing models/NeoVerse/reconstructor.ckpt" >&2; exit 1; }

# --- Train ----------------------------------------------------------------
# The L1 metric anchor is on in the ablation config (loss_weights.hand_depth_anchor).
# Watch W&B: train/hand_depth_residual_m should fall toward the 0.02 m margin.
python3 -m scripts.train_hand_head --config "${CONFIG}"

echo "Finished   : $(date)"
