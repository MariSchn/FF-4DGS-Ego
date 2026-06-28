#!/bin/bash
#SBATCH --account=3dv
#SBATCH --partition=jobs
#SBATCH --gpus=gb10:1
# (b) Cyrus direction b: frozen GS backbone + gs_head, train only a small ScaleHead
# (+ hand head) so s*gs_depth matches the metric MANO hand. Never run before. Frozen
# backbone forward (no_grad) -> fast. Early validation (val_every=20) shows whether
# scale_residual_m decreases. Checkpoints -> node-local $TMPDIR.
set -uo pipefail
cd /home/dmonopoli/FF-4DGS-Ego || exit 2
source venv_gb10/bin/activate
OUT="${TMPDIR:-/tmp}/p3b_${SLURM_JOB_ID}"; mkdir -p "$OUT"
echo "=== P3b SCALEHEAD job $SLURM_JOB_ID @ $(date) on $(hostname) ==="
echo "=== out=$OUT  arch=$(uname -m) ==="
python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
echo "=== TRAIN START @ $(date) ==="
python -m scripts.train_hand_head --config configs/exp_p3_scalehead.yaml \
  training.output_dir="$OUT" training.val_every=20
echo "=== P3b rc=$? COMPLETE @ $(date) ==="
