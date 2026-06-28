#!/bin/bash
# ---------------------------------------------------------------------------
# HOI4D backbone-compatibility smoke: does the frozen backbone give sane
# gs_depth on HOI4D RGB? (~1-2 min). Uses #SBATCH --output to /work/scratch
# (the exec>tee hack produced empty files; this writes the log directly).
#   sbatch hoi4d_smoke_gb10.sh
# ---------------------------------------------------------------------------
#SBATCH --job-name=ff4dgs-hoi4dsmk
#SBATCH --account=3dv
#SBATCH --gpus=gb10:1
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/work/scratch/dmonopoli/joblogs/%j_hoi4dsmoke.out
#SBATCH --error=/work/scratch/dmonopoli/joblogs/%j_hoi4dsmoke.out

set -euo pipefail
cd /home/dmonopoli/FF-4DGS-Ego
source venv_gb10/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1

SEQ=/work/scratch/dmonopoli/hoi4d/ZY20210800001_H1_C11_N07_S185_s02_T2/images
echo "=== HOI4D backbone smoke $(date) ==="
python3 -m scripts.hoi4d_backbone_smoke \
  --config configs/exp_p3_gtdepth_unfreeze.yaml \
  --rgb "$SEQ" --frames 16 --out /work/scratch/dmonopoli/hoi4d_smoke
echo "=== done $(date) ==="
