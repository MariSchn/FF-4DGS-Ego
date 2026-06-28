#!/bin/bash
# Render two more P0001 moments (f2000, f0200) in Marian's gsplat style.
#SBATCH --job-name=ff4dgs-newsc
#SBATCH --account=3dv
#SBATCH --gpus=gb10:1
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#SBATCH --output=/work/scratch/dmonopoli/joblogs/%j_newsc.out
#SBATCH --error=/work/scratch/dmonopoli/joblogs/%j_newsc.out
set -uo pipefail
cd /home/dmonopoli/FF-4DGS-Ego
source venv_gb10/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
BASE=/work/courses/3dv/team25/outputs/hand_alignment
for D in P0001_10a27bf7_f2000_undist_f609 P0001_10a27bf7_f0200_undist_f609; do
  echo "=== render $D $(date) ==="
  python3 scripts/hand_alignment/render_alignment_3d.py \
    --output_dir "$BASE/$D" --out_subdir report_3d_hires --mode both \
    --res 1080 --hero 8 --bg 1,1,1 || echo "FAILED $D"
  ls "$BASE/$D"/report_3d_hires/ 2>/dev/null
done
echo "=== done $(date) ==="
