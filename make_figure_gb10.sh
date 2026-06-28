#!/bin/bash
# Renderable-Gaussian hero figure: GT metric hand composited into the feedforward
# 4D-Gaussian scene (the metric-coupling visualization). align -> gsplat render.
#SBATCH --job-name=ff4dgs-fig
#SBATCH --account=3dv
#SBATCH --gpus=gb10:1
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=/work/scratch/dmonopoli/joblogs/%j_fig.out
#SBATCH --error=/work/scratch/dmonopoli/joblogs/%j_fig.out
set -uo pipefail
cd /home/dmonopoli/FF-4DGS-Ego
source venv_gb10/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
SEQ="${1:-P0001_10a27bf7}"
OUT=/work/scratch/dmonopoli/fig_align_$SEQ
echo "=== align $SEQ $(date) ==="
python3 scripts/hand_alignment/align_hand_to_neoverse.py \
  --data_root /work/scratch/dmonopoli/hot3d_data/hot3d_aria/sequences \
  --sequence_id "$SEQ" --frame_start 0 --num_frames 16 \
  --undistort --undistort_focal 280 \
  --height 336 --width 336 --output_dir "$OUT" || { echo "ALIGN FAILED"; exit 1; }
echo "=== render $(date) ==="
python3 scripts/hand_alignment/render_alignment_3d.py \
  --output_dir "$OUT" --mode both --res 1080 --hero 6 --bg 1,1,1
echo "=== done $(date) ==="
ls -la "$OUT"/report_3d/ 2>/dev/null
