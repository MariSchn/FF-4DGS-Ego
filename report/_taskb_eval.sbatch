#!/bin/bash
#SBATCH --job-name=hires_evalB
#SBATCH --account=3dv
#SBATCH --partition=jobs
#SBATCH --gpus=gb10:1
#SBATCH --time=01:00:00
#SBATCH --output=/home/dmonopoli/hires_evalB_%j.out
#SBATCH --error=/home/dmonopoli/hires_evalB_%j.out
# ---------------------------------------------------------------------------
# Evaluates the FULL hires hand model trained on subjects 1+2+3.
# Copies pre-packed subject4 npz files from scratch to node-local $TMPDIR.
# ---------------------------------------------------------------------------
set -uo pipefail
echo "=== EVAL-B JOB $SLURM_JOB_ID on $(hostname) @ $(date) ==="
cd /home/dmonopoli/FF-4DGS-Ego
source venv_gb10/bin/activate
echo "PYTHON: $(which python)  ARCH: $(uname -m)"
python -c "import torch,torchvision;print('torch',torch.__version__,'tv',torchvision.__version__,'cuda',torch.cuda.is_available())"

WORK="${TMPDIR:-/tmp}/dm_evalB_$SLURM_JOB_ID"
PACK="$WORK/h2o_hires"
mkdir -p "$PACK"
echo "=== node-local workdir: $WORK ===" ; df -h "$WORK" | tail -1

echo "=== COPY pre-packed subject4 files @ $(date) ==="
cp -r /work/scratch/dmonopoli/h2o_packed_hires/subject4* "$PACK/"
echo "=== COPY done @ $(date) ==="
S4=$(ls "$PACK"/subject4*.npz 2>/dev/null | wc -l)
echo "copied npz: subject4=$S4" ; du -sh "$PACK"

HIRES_CKPT=/work/scratch/dmonopoli/checkpoints/h2o_hand_hires_full/best_cmpjpe.pt
BASE_CKPT=/work/scratch/dmonopoli/checkpoints/h2o_hand/best_cmpjpe.pt

HCFG="$WORK/exp_h2o_hand_hires.yaml"
sed "s|__SET_AT_JOB_TIME__|$PACK|" configs/exp_h2o_hand_hires.yaml > "$HCFG"
BCFG="$WORK/exp_h2o_hand_base.yaml"
sed "s|/work/scratch/dmonopoli/h2o_packed|$PACK|" configs/exp_h2o_hand.yaml > "$BCFG"

echo ""
echo "############################################################"
echo "### EVAL HIRES ckpt (PA-MPJPE, 21-joint, subject4, --hires)"
echo "###   ckpt: $HIRES_CKPT"
echo "############################################################"
if [ -f "$HIRES_CKPT" ]; then
  python -m scripts.eval_cmpjpe --config "$HCFG" --ckpt "$HIRES_CKPT" \
    --h2o "$PACK" --subject subject4 --joints21 --hires --limit 400
  echo "=== EVAL HIRES rc=$? ==="
else
  echo "=== ERROR: hires ckpt missing at $HIRES_CKPT ==="
fi

echo ""
echo "############################################################"
echo "### EVAL BASELINE 224px trained ckpt (PA-MPJPE, 21-joint, subject4, NO --hires)"
echo "###   ckpt: $BASE_CKPT"
echo "############################################################"
if [ -f "$BASE_CKPT" ]; then
  python -m scripts.eval_cmpjpe --config "$BCFG" --ckpt "$BASE_CKPT" \
    --h2o "$PACK" --subject subject4 --joints21 --limit 400
  echo "=== EVAL BASELINE rc=$? ==="
else
  echo "=== ERROR: baseline ckpt missing at $BASE_CKPT ==="
fi

echo "=== EVAL-B JOB $SLURM_JOB_ID COMPLETE @ $(date) ==="
