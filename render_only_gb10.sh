#!/bin/bash
# Render an already-exported figure dir with gsplat. Arg 1 = output_dir.
#SBATCH --job-name=ff4dgs-rend
#SBATCH --account=3dv
#SBATCH --gpus=gb10:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/work/scratch/dmonopoli/joblogs/%j_rend.out
#SBATCH --error=/work/scratch/dmonopoli/joblogs/%j_rend.out
set -uo pipefail
cd /home/dmonopoli/FF-4DGS-Ego
source venv_gb10/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
OUT="${1:-/work/scratch/dmonopoli/fig_h2o_clip0}"
echo "=== render $OUT $(date) ==="
python3 scripts/hand_alignment/render_alignment_3d.py \
  --output_dir "$OUT" --mode both --res 1080 --hero 4 --bg 1,1,1
echo "=== done $(date) ==="
ls -la "$OUT"/report_3d/ 2>/dev/null
