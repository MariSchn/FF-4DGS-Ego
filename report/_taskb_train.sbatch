#!/bin/bash
#SBATCH --job-name=hires_full
#SBATCH --account=3dv
#SBATCH --partition=jobs
#SBATCH --gpus=gb10:1
#SBATCH --time=08:00:00
#SBATCH --output=/home/dmonopoli/hires_full_%j.out
#SBATCH --error=/home/dmonopoli/hires_full_%j.out
# ---------------------------------------------------------------------------
# TASK (b): FULL hires train on subject1+2+3 (train), eval subject4 (held-out).
#   - Pack H2O subject1,2,3 (train) + subject4 (test) at --hand_crop_px 192 to
#     NODE-LOCAL $TMPDIR (scratch at quota; /work writes can silently cancel).
#   - Train train_h2o_hand with model.hires_hand=true to the 8h cap.
#   - Save best ckpt to $HOME/ckpt_h2o_hand_hires_full (distinct dir, no clobber).
#   - Eval PA-MPJPE (21-joint, Procrustes) of trained hires ckpt AND the 224px
#     trained baseline on the SAME subject4 clips -> fair full-data hires number.
# ---------------------------------------------------------------------------
set -uo pipefail
echo "=== HIRES-FULL JOB $SLURM_JOB_ID on $(hostname) @ $(date) ==="
cd /home/dmonopoli/FF-4DGS-Ego
source venv_gb10/bin/activate
echo "PYTHON: $(which python)  ARCH: $(uname -m)"
python -c "import torch,torchvision;print('torch',torch.__version__,'tv',torchvision.__version__,'cuda',torch.cuda.is_available())"

WORK="${TMPDIR:-/tmp}/dm_hiresfull_$SLURM_JOB_ID"
PACK="$WORK/h2o_hires"
mkdir -p "$PACK"
echo "=== node-local workdir: $WORK ===" ; df -h "$WORK" | tail -1

U="${H2O_USER}" ; P="${H2O_PASS}"
pack_subject () {
  local subj="$1" ; local url="https://h2odataset.ethz.ch/data/dataset/${subj}_ego_v1_1.tar.gz"
  echo "=== PACK $subj @ $(date) -> $PACK ==="
  curl -sL -u "$U:$P" "$url" \
    | python3 -m scripts.pack_h2o --out "$PACK" --res 224 --hand_crop_px 192
  echo "=== PACK $subj done @ $(date) ==="
}
# subject1+2+3 = train (full cross-subject train set), subject4 = held-out test.
pack_subject subject1
pack_subject subject2
pack_subject subject3
pack_subject subject4
echo "=== PACK SUMMARY ===" ; du -sh "$PACK"
S1=$(ls "$PACK"/subject1*.npz 2>/dev/null | wc -l)
S2=$(ls "$PACK"/subject2*.npz 2>/dev/null | wc -l)
S3=$(ls "$PACK"/subject3*.npz 2>/dev/null | wc -l)
S4=$(ls "$PACK"/subject4*.npz 2>/dev/null | wc -l)
echo "packed npz: s1=$S1 s2=$S2 s3=$S3 s4(test)=$S4"
if [ "$S1" -eq 0 ] || [ "$S2" -eq 0 ] || [ "$S3" -eq 0 ] || [ "$S4" -eq 0 ]; then
  echo "=== FATAL: missing a split (s1=$S1 s2=$S2 s3=$S3 s4=$S4). Aborting. ==="
  exit 11
fi

# Config: data_root -> node-local pack; output_dir -> distinct $HOME dir (no clobber
# of the subset run's $HOME/ckpt_h2o_hand_hires). test_subject stays subject4 so the
# train/val split holds out subject4 and trains on subject1+2+3.
CFG="$WORK/exp_h2o_hand_hires_full.yaml"
sed -e "s|__SET_AT_JOB_TIME__|$PACK|" \
    -e "s|/home/dmonopoli/ckpt_h2o_hand_hires|/home/dmonopoli/ckpt_h2o_hand_hires_full|" \
    configs/exp_h2o_hand_hires.yaml > "$CFG"
echo "=== data_root  -> $(grep data_root $CFG) ==="
echo "=== output_dir -> $(grep output_dir $CFG) ==="
echo "=== test_subj  -> $(grep test_subject $CFG) ==="

echo "=== TRAIN @ $(date) ==="
python -m scripts.train_h2o_hand --config "$CFG"
TRAIN_RC=$?
echo "=== TRAIN done @ $(date)  rc=$TRAIN_RC ==="

CKDIR=/home/dmonopoli/ckpt_h2o_hand_hires_full
BEST="$CKDIR/best_cmpjpe.pt"
echo "=== saved checkpoints ===" ; ls -la "$CKDIR" 2>&1

BASE_CKPT=/work/scratch/dmonopoli/checkpoints/h2o_hand/best_cmpjpe.pt
BCFG="$WORK/exp_h2o_hand_base.yaml"
sed "s|/work/scratch/dmonopoli/h2o_packed|$PACK|" configs/exp_h2o_hand.yaml > "$BCFG"

if [ -f "$BEST" ]; then
  echo ""
  echo "### EVAL HIRES-FULL ckpt (PA-MPJPE, 21-joint, subject4, --hires) @ $(date) ###"
  python -m scripts.eval_cmpjpe --config "$CFG" --ckpt "$BEST" \
    --h2o "$PACK" --subject subject4 --joints21 --hires --limit 400
  echo "=== EVAL HIRES-FULL rc=$? ==="
fi
echo ""
echo "### EVAL BASELINE 224px trained ckpt (PA-MPJPE, 21-joint, subject4, NO --hires) @ $(date) ###"
python -m scripts.eval_cmpjpe --config "$BCFG" --ckpt "$BASE_CKPT" \
  --h2o "$PACK" --subject subject4 --joints21 --limit 400
echo "=== EVAL BASELINE rc=$? ==="

echo "=== HIRES-FULL JOB $SLURM_JOB_ID COMPLETE @ $(date) ==="
