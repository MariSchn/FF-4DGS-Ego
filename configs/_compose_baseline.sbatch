#!/bin/bash
#SBATCH --job-name=compose_bl
#SBATCH --account=3dv
#SBATCH --gpus=gb10:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/home/dmonopoli/joblogs/compose_baseline_%j.out
#SBATCH --error=/home/dmonopoli/joblogs/compose_baseline_%j.err

set -uo pipefail
cd /home/dmonopoli/FF-4DGS-Ego
source /work/scratch/dmonopoli/venv_gb10/bin/activate

echo "=== COMPOSE + SCORE BASELINES @ $(date) ==="

# 1. Compose world-space baseline for HaMeR + SLAM trajectory
python scripts/build_slam_baseline.py \
  --cam_pred_dir /home/dmonopoli/hamer_gtbox_preds \
  --slam_pred_dir /home/dmonopoli/wilor_gtbox_preds \
  --out_dir /home/dmonopoli/hamer_slam_preds

# 2. Score HaMeR + SLAM
python -m scripts.eval_worldspace_baseline \
  --data_root /home/dmonopoli/hoi4d_test \
  --pred_dir /home/dmonopoli/hamer_slam_preds \
  --segment_len 128 \
  --wa_short 16 \
  --out /home/dmonopoli/results/hamer_slam_eval.json

# 3. Compose world-space baseline for WiLoR + SLAM trajectory
python scripts/build_slam_baseline.py \
  --cam_pred_dir /home/dmonopoli/wilor_gtbox_preds \
  --slam_pred_dir /home/dmonopoli/wilor_gtbox_preds \
  --out_dir /home/dmonopoli/wilor_slam_preds

# 4. Score WiLoR + SLAM
python -m scripts.eval_worldspace_baseline \
  --data_root /home/dmonopoli/hoi4d_test \
  --pred_dir /home/dmonopoli/wilor_slam_preds \
  --segment_len 128 \
  --wa_short 16 \
  --out /home/dmonopoli/results/wilor_slam_eval.json

echo "=== COMPOSE + SCORE BASELINES DONE @ $(date) ==="
