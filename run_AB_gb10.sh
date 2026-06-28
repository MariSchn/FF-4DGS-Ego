#!/bin/bash
# (A) Re-render Marian's Figure-5 assets in his own gsplat style (high-res hero set),
# (B) PA-MPJPE eval of the finished v2 cosine fine-tune. One gb10 job.
#SBATCH --job-name=ff4dgs-AB
#SBATCH --account=3dv
#SBATCH --gpus=gb10:1
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/work/scratch/dmonopoli/joblogs/%j_AB.out
#SBATCH --error=/work/scratch/dmonopoli/joblogs/%j_AB.out
set -uo pipefail
cd /home/dmonopoli/FF-4DGS-Ego
source venv_gb10/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
D=/work/courses/3dv/team25/outputs/hand_alignment/P0001_10a27bf7_f1000_undist_f609

echo "=== (A) gsplat render of Marian's assets (his exact style) $(date) ==="
python3 scripts/hand_alignment/render_alignment_3d.py \
  --output_dir "$D" --out_subdir report_3d_hires --mode both \
  --res 1080 --hero 8 --bg 1,1,1 || echo "RENDER FAILED (rc=$?)"
echo "--- render outputs ---"
ls -la "$D"/report_3d_hires/ 2>/dev/null

echo "=== (B) 21-joint PA-MPJPE eval of v2 fine-tune $(date) ==="
python3 -m scripts.eval_cmpjpe \
  --config configs/exp_h2o_hand.yaml \
  --ckpt /work/scratch/dmonopoli/checkpoints/h2o_hand_v2/best_cmpjpe.pt \
  --h2o /work/scratch/dmonopoli/h2o_packed --subject subject4 \
  --joints21 --limit 400 || echo "EVAL FAILED (rc=$?)"
echo "=== done $(date) ==="
