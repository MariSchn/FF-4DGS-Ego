#!/bin/bash
# Projectaria-free H2O figure: feedforward Gaussians + PREDICTED metric hand.
# export (forward GS+hand, scale solve) -> gsplat render. Arg 1 = clip index.
#SBATCH --job-name=ff4dgs-figh2o
#SBATCH --account=3dv
#SBATCH --gpus=gb10:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/work/scratch/dmonopoli/joblogs/%j_figh2o.out
#SBATCH --error=/work/scratch/dmonopoli/joblogs/%j_figh2o.out
set -uo pipefail
cd /home/dmonopoli/FF-4DGS-Ego
source venv_gb10/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
CLIP="${1:-0}"
OUT=/work/scratch/dmonopoli/fig_h2o_clip${CLIP}
echo "=== export H2O figure (clip $CLIP) $(date) ==="
python3 -m scripts.export_h2o_figure \
  --config configs/exp_h2o_hand.yaml \
  --ckpt /work/scratch/dmonopoli/checkpoints/h2o_hand/best_cmpjpe.pt \
  --h2o /work/scratch/dmonopoli/h2o_packed --subject subject4 \
  --clip_index "$CLIP" --output_dir "$OUT" || { echo "EXPORT FAILED"; exit 1; }
echo "=== render $(date) ==="
python3 scripts/hand_alignment/render_alignment_3d.py \
  --output_dir "$OUT" --mode both --res 1080 --hero 4 --bg 1,1,1
echo "=== done $(date) ==="
ls -la "$OUT"/report_3d/ 2>/dev/null
