#!/bin/bash
#SBATCH --job-name=prep_hoi4d
#SBATCH --account=3dv
#SBATCH --partition=jobs
#SBATCH --gpus=gb10:1
#SBATCH --time=04:00:00
#SBATCH --output=/home/dmonopoli/prep_hoi4d_%j.out
#SBATCH --error=/home/dmonopoli/prep_hoi4d_%j.out
# ---------------------------------------------------------------------------
# Preprocesses HOI4D sequence images, depths, camera poses, and hand poses.
# Unzips annotation and hand pose packages to node-local $TMPDIR on-the-fly
# to avoid shared scratch inode quota limits.
# ---------------------------------------------------------------------------
set -uo pipefail
echo "=== PREPROCESS JOB $SLURM_JOB_ID on $(hostname) @ $(date) ==="
cd /home/dmonopoli/FF-4DGS-Ego
source venv_gb10/bin/activate
echo "PYTHON: $(which python)  ARCH: $(uname -m)"

# 1) Setup node-local directories
mkdir -p "$TMPDIR/annotations"
mkdir -p "$TMPDIR/hand_pose"

# 2) Unzip files on-the-fly to node-local SSD (keeps scratch inode count safe!)
echo "=== Unzipping annotations to $TMPDIR ==="
unzip -q /work/scratch/dmonopoli/HOI4D_annotations_poses_only.zip -d "$TMPDIR/annotations"
echo "=== Unzipping hand poses to $TMPDIR ==="
unzip -q /work/scratch/dmonopoli/HOI4D_Hand_pose.zip -d "$TMPDIR/hand_pose"
echo "=== Unzip completed ==="

# 3) Loop over all sequences and run preprocessing
for seq_dir in /work/scratch/dmonopoli/hoi4d/ZY2021*; do
    [ -d "$seq_dir" ] || continue
    seq_folder=$(basename "$seq_dir")
    nested_seq=$(echo "$seq_folder" | sed 's|_|/|g')
    echo ""
    echo "############################################################"
    echo "### Preprocessing sequence: $nested_seq"
    echo "############################################################"
    
    python -m scripts.preprocessing.preprocess_hoi4d \
      --hoi4d_root "$TMPDIR/annotations/HOI4D_annotations" \
      --rgb_root /work/scratch/dmonopoli/hoi4d \
      --handpose_root "$TMPDIR/hand_pose/Hand_pose/handpose_right_hand" \
      --seq "$nested_seq" \
      --out /work/scratch/dmonopoli/hoi4d_pp \
      --res 224
done

echo "=== PREPROCESSING JOB $SLURM_JOB_ID COMPLETE @ $(date) ==="
