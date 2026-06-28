#!/bin/bash
#SBATCH --job-name=ff4dgs-eval2
#SBATCH --account=3dv
#SBATCH --gpus=gb10:1
#SBATCH --mem=32G
#SBATCH --time=00:50:00
#SBATCH --output=/work/scratch/dmonopoli/joblogs/%j_eval2.out
#SBATCH --error=/work/scratch/dmonopoli/joblogs/%j_eval2.out
set -uo pipefail
cd /home/dmonopoli/FF-4DGS-Ego
source venv_gb10/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
for NAME in h2o_hand h2o_hand_v2; do
  echo "=== eval $NAME (21-joint, subject4) $(date) ==="
  python3 -m scripts.eval_cmpjpe --config configs/exp_h2o_hand.yaml \
    --ckpt /work/scratch/dmonopoli/checkpoints/$NAME/best_cmpjpe.pt \
    --h2o /work/scratch/dmonopoli/h2o_packed --subject subject4 --joints21 --limit 400 || echo "FAILED $NAME"
done
echo "=== done $(date) ==="
