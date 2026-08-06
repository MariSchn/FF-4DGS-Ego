#!/bin/bash
#SBATCH --job-name=eval_reg_correct
#SBATCH --account=3dv
#SBATCH --partition=jobs
#SBATCH --gpus=gb10:1
#SBATCH --time=00:30:00
#SBATCH --output=/home/dmonopoli/eval_reg_correct_%j.out
#SBATCH --error=/home/dmonopoli/eval_reg_correct_%j.out

set -uo pipefail
echo "=== EVAL REGISTRATION CORRECT JOB $SLURM_JOB_ID on $(hostname) @ $(date) ==="
cd /home/dmonopoli/FF-4DGS-Ego
source venv_gb10/bin/activate
echo "PYTHON: $(which python)  ARCH: $(uname -m)"

WORK="${TMPDIR:-/tmp}/dm_eval_reg_correct_$SLURM_JOB_ID"
PACK="$WORK/h2o_hires"
mkdir -p "$PACK"

echo "=== COPY pre-packed subject4 files @ $(date) ==="
cp -r /work/scratch/dmonopoli/h2o_packed_hires/subject4* "$PACK/"
echo "=== COPY done @ $(date) ==="
S4=$(ls "$PACK"/subject4*.npz 2>/dev/null | wc -l)
echo "copied npz: subject4=$S4" ; du -sh "$PACK"

REG_CKPT=/work/scratch/dmonopoli/checkpoints/p3_registration_fast/best_mpjpe.pt
HCFG="configs/exp_p3_registration_fast.yaml"

echo ""
echo "############################################################"
echo "### EVAL REGISTRATION CKPT CORRECT (PA-MPJPE, 21-joint, subject4, NO --hires)"
echo "###   ckpt: $REG_CKPT"
echo "############################################################"
if [ -f "$REG_CKPT" ]; then
  python -m scripts.eval_cmpjpe --config "$HCFG" --ckpt "$REG_CKPT" \
    --h2o "$PACK" --subject subject4 --joints21 --limit 400
  echo "=== EVAL REG rc=$? ==="
else
  echo "=== ERROR: registration ckpt missing at $REG_CKPT ==="
fi

echo "=== CLEANUP @ $(date) ==="
rm -rf "$WORK"
echo "=== EVAL REG CORRECT JOB $SLURM_JOB_ID COMPLETE @ $(date) ==="
