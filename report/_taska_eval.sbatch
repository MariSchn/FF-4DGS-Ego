#!/bin/bash
#SBATCH --job-name=hires_evalA
#SBATCH --account=3dv
#SBATCH --partition=jobs
#SBATCH --gpus=gb10:1
#SBATCH --time=02:00:00
#SBATCH --output=/home/dmonopoli/hires_evalA_%j.out
#SBATCH --error=/home/dmonopoli/hires_evalA_%j.out
# ---------------------------------------------------------------------------
# TASK (a) -- DECISIVE: hires-vs-baseline PA on IDENTICAL subject4 clips.
#   1. Pack H2O subject4 at --hand_crop_px 192 -> NODE-LOCAL $TMPDIR (scratch at quota).
#   2. Eval HIRES ckpt ($HOME/ckpt_h2o_hand_hires/best_cmpjpe.pt) with --hires.
#   3. Eval BASELINE 224px trained ckpt (.../h2o_hand/best_cmpjpe.pt) WITHOUT --hires.
#   Both on the SAME subject4 clips, --joints21 Procrustes. Report PA/C-MPJPE/WRIST.
#   Question: does hires beat the baseline's clean PA 46.76?
# ---------------------------------------------------------------------------
set -uo pipefail
echo "=== EVAL-A JOB $SLURM_JOB_ID on $(hostname) @ $(date) ==="
cd /home/dmonopoli/FF-4DGS-Ego
source venv_gb10/bin/activate
echo "PYTHON: $(which python)  ARCH: $(uname -m)"
python -c "import torch,torchvision;print('torch',torch.__version__,'tv',torchvision.__version__,'cuda',torch.cuda.is_available())"

WORK="${TMPDIR:-/tmp}/dm_evalA_$SLURM_JOB_ID"
PACK="$WORK/h2o_hires"
mkdir -p "$PACK"
echo "=== node-local workdir: $WORK ===" ; df -h "$WORK" | tail -1

U="${H2O_USER}" ; P="${H2O_PASS}"
echo "=== PACK subject4 (hand_crop_px 192) @ $(date) -> $PACK ==="
curl -sL -u "$U:$P" "https://h2odataset.ethz.ch/data/dataset/subject4_ego_v1_1.tar.gz" \
  | python3 -m scripts.pack_h2o --out "$PACK" --res 224 --hand_crop_px 192
echo "=== PACK subject4 done @ $(date) ==="
S4=$(ls "$PACK"/subject4*.npz 2>/dev/null | wc -l)
echo "packed npz: subject4=$S4" ; du -sh "$PACK"
if [ "$S4" -eq 0 ]; then echo "=== FATAL: no subject4 npz packed. Aborting. ==="; exit 11; fi

HIRES_CKPT=/home/dmonopoli/ckpt_h2o_hand_hires/best_cmpjpe.pt
BASE_CKPT=/work/scratch/dmonopoli/checkpoints/h2o_hand/best_cmpjpe.pt

# data_root in the configs is unused by eval (--h2o drives it), but sed it anyway for hygiene.
HCFG="$WORK/exp_h2o_hand_hires.yaml"
sed "s|__SET_AT_JOB_TIME__|$PACK|" configs/exp_h2o_hand_hires.yaml > "$HCFG"
BCFG="$WORK/exp_h2o_hand_base.yaml"
sed "s|/work/scratch/dmonopoli/h2o_packed|$PACK|" configs/exp_h2o_hand.yaml > "$BCFG"

echo ""
echo "############################################################"
echo "### EVAL HIRES ckpt (PA-MPJPE, 21-joint, subject4, --hires) @ $(date)"
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
echo "### EVAL BASELINE 224px trained ckpt (PA-MPJPE, 21-joint, subject4, NO --hires) @ $(date)"
echo "###   ckpt: $BASE_CKPT  (clean baseline PA was 46.76)"
echo "############################################################"
if [ -f "$BASE_CKPT" ]; then
  python -m scripts.eval_cmpjpe --config "$BCFG" --ckpt "$BASE_CKPT" \
    --h2o "$PACK" --subject subject4 --joints21 --limit 400
  echo "=== EVAL BASELINE rc=$? ==="
else
  echo "=== ERROR: baseline ckpt missing at $BASE_CKPT ==="
fi

echo "=== EVAL-A JOB $SLURM_JOB_ID COMPLETE @ $(date) ==="
