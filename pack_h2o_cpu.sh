#!/bin/bash
# Stream-download + pack H2O subject1 into .npz on the CPU partition (concurrent
# with GPU training). Creds inherited from submit env ($H2O_USER/$H2O_PASS).
#SBATCH --job-name=ff4dgs-h2opack
#SBATCH --account=3dv
#SBATCH --partition=interactive-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/work/scratch/dmonopoli/joblogs/%j_h2opack.out
#SBATCH --error=/work/scratch/dmonopoli/joblogs/%j_h2opack.out
set -uo pipefail
cd /home/dmonopoli/FF-4DGS-Ego
echo "=== H2O pack start $(date) ==="
curl -sL -u "$H2O_USER:$H2O_PASS" https://h2odataset.ethz.ch/data/dataset/subject1_ego_v1_1.tar.gz \
  | python3 scripts/pack_h2o.py --out /work/scratch/dmonopoli/h2o_packed --res 224
echo "=== packed files: $(ls /work/scratch/dmonopoli/h2o_packed 2>/dev/null | wc -l) | $(date) ==="
