#!/bin/bash
#SBATCH --job-name=ff4dgs-b2dbg
#SBATCH --account=3dv
#SBATCH --gpus=gb10:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
set -uo pipefail
cd "$HOME/FF-4DGS-Ego"
source venv_gb10/bin/activate
export B2_DEBUG=1 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "B2 DEBUG run (a01 recovered ckpt, 6 clips, 4 seqs)"
python3 -m scripts.eval_scene_metric_gt \
  --config configs/exp_p2_pinhole_warmstart_a01.yaml \
  --checkpoint checkpoints/p2_warmstart_a01/best_mpjpe.pt \
  --objects_dir /work/scratch/dmonopoli/hot3d_assets --num_clips 6 --max_seqs 4 || true
echo "B2 DEBUG done"
