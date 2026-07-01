#!/bin/bash
#SBATCH --job-name=eval_reg
#SBATCH --account=3dv
#SBATCH --partition=jobs
#SBATCH --gpus=gb10:1
#SBATCH --time=00:30:00
#SBATCH --output=/home/dmonopoli/eval_reg_%j.out
#SBATCH --error=/home/dmonopoli/eval_reg_%j.out
# ---------------------------------------------------------------------------
# Evaluates the Step 50 checkpoint from the hand-scene registration job.
# Copies pre-packed subject4 npz files from scratch to node-local $TMPDIR.
# ---------------------------------------------------------------------------
set -uo pipefail
echo "=== EVAL REGISTRATION JOB $SLURM_JOB_ID on $(hostname) @ $(date) ==="
cd /home/dmonopoli/FF-4DGS-Ego
source venv_gb10/bin/activate
echo "PYTHON: $(which python)  ARCH: $(uname -m)"

WORK="${TMPDIR:-/tmp}/dm_eval_reg_$SLURM_JOB_ID"
PACK="$WORK/h2o_hires"
mkdir -p "$PACK"

echo "=== COPY pre-packed subject4 files @ $(date) ==="
cp -r /work/scratch/dmonopoli/h2o_packed_hires/subject4* "$PACK/"
echo "=== COPY done @ $(date) ==="
S4=$(ls "$PACK"/subject4*.npz 2>/dev/null | wc -l)
echo "copied npz: subject4=$S4" ; du -sh "$PACK"

REG_CKPT=/work/scratch/dmonopoli/checkpoints/p3_registration_fast/best_mpjpe.pt
HCFG="$WORK/exp_h2o_hand_hires.yaml"
sed "s|__SET_AT_JOB_TIME__|$PACK|" configs/exp_h2o_hand_hires.yaml > "$HCFG"

echo ""
echo "############################################################"
echo "### EVAL REGISTRATION CKPT (PA-MPJPE, 21-joint, subject4, --hires)"
echo "###   ckpt: $REG_CKPT"
echo "############################################################"
if [ -f "$REG_CKPT" ]; then
  python -m scripts.eval_cmpjpe --config "$HCFG" --ckpt "$REG_CKPT" \
    --h2o "$PACK" --subject subject4 --joints21 --hires --limit 400
  echo "=== EVAL REG rc=$? ==="
else
  echo "=== ERROR: registration ckpt missing at $REG_CKPT ==="
fi

echo "=== EVAL REG JOB $SLURM_JOB_ID COMPLETE @ $(date) ==="
