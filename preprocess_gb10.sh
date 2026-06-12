#!/bin/bash
# ---------------------------------------------------------------------------
# Pinhole-rectify HOT3D sequences on a gb10 node (P2 prep).
#
# Runs scripts/preprocessing/preprocess_undistort.py with the exact params of
# the best historical run (pinhole_f609_rf1.5_res224, 50.08mm MPJPE): fisheye
# -> pinhole at f=609.78, bbox cache rf=1.5 @ res 224. Defaults to the 5
# sequences the P1 experiments train on (sorted-first-5 of the data root).
#
# The gb10 venv already has projectaria_tools / decord / cv2 / trimesh; the
# job is CPU-bound (~10-20 min per sequence). Chain between training jobs:
#   sbatch --dependency=afterany:<train_jobid> preprocess_gb10.sh
# ---------------------------------------------------------------------------
#SBATCH --job-name=ff4dgs-preproc
#SBATCH --account=3dv
#SBATCH --gpus=gb10:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail

SRC="${1:-/work/courses/3dv/team25/data/hot3d_aria/sequences}"
DST="${2:-/work/courses/3dv/team25/data/hot3d_aria/preprocessed_pinhole_f609}"
SEQS="${3:-P0001_10a27bf7,P0001_15c4300c,P0001_23fa0ee8,P0001_4bf4e21a,P0001_550ea2ac}"
mkdir -p logs

echo "=================================================="
echo "Job        : ${SLURM_JOB_ID:-?} on $(hostname)"
echo "Source     : ${SRC}"
echo "Dest       : ${DST}"
echo "Sequences  : ${SEQS}"
echo "Started    : $(date)"
echo "=================================================="

# --- Environment (GB10-specific venv from prep_venv.sh) -------------------
if [[ ! -f venv_gb10/bin/activate ]]; then
  echo "ERROR: venv_gb10 not found. Run ./prep_venv.sh first." >&2
  exit 1
fi
source venv_gb10/bin/activate
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

python3 scripts/preprocessing/preprocess_undistort.py \
  --source_root "${SRC}" \
  --dest_root   "${DST}" \
  --focal 609.78 --rescale_factor 1.5 --resolution 224 \
  --sequences "${SEQS}"

echo "Finished   : $(date)"
