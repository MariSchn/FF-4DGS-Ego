#!/bin/bash
#SBATCH --account=3dv
#SBATCH --partition=jobs
#SBATCH --gpus=gb10:1
# (a) Cyrus direction a: partial unfreeze (last 4 frame+global blocks) + GT object
# depth supervision on HOT3D. Early validation (val_every=8) so the object-depth
# verdict (does obj_depth_residual_m drop below the 0.62 m frozen baseline?) lands
# within ~30 min even though a full epoch is ~65 h on this card. Checkpoints ->
# node-local $TMPDIR ($HOME inode-dead, /work per-user quota full).
set -uo pipefail
cd /home/dmonopoli/FF-4DGS-Ego || exit 2
source venv_gb10/bin/activate
OUT="${TMPDIR:-/tmp}/p3a_${SLURM_JOB_ID}"; mkdir -p "$OUT"
echo "=== P3a UNFREEZE+GTDEPTH job $SLURM_JOB_ID @ $(date) on $(hostname) ==="
echo "=== out=$OUT  arch=$(uname -m) ==="
python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
echo "=== TRAIN START @ $(date) ==="
python -m scripts.train_hand_head --config configs/exp_p3_gtdepth_unfreeze.yaml \
  training.output_dir="$OUT" training.val_every=8
echo "=== P3a rc=$? COMPLETE @ $(date) ==="
