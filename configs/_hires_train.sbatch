#!/bin/bash
#SBATCH --job-name=hires_train
#SBATCH --account=3dv
#SBATCH --partition=jobs
#SBATCH --gpus=gb10:1
#SBATCH --time=08:00:00
#SBATCH --output=/home/dmonopoli/hires_train_%j.out
#SBATCH --error=/home/dmonopoli/hires_train_%j.out
# ---------------------------------------------------------------------------
# Self-contained Phase-2 hires-hand job: pack -> train -> eval -> save ckpt.
#   - Packs H2O subject1 (train) + subject4 (test) at --hand_crop_px 192 to
#     NODE-LOCAL $TMPDIR (scratch is at quota; /work writes can silently cancel).
#   - Trains train_h2o_hand with model.hires_hand=true on the node-local pack.
#   - Saves only the best checkpoint to scratch (output_dir in the config).
#   - Evals PA-MPJPE (21-joint, Procrustes) of the trained hires ckpt AND the
#     224px warm-start baseline on the SAME subject4 clips -> apples-to-apples.
# ---------------------------------------------------------------------------
set -uo pipefail
echo "=== HIRES JOB $SLURM_JOB_ID on $(hostname) @ $(date) ==="
cd /home/dmonopoli/FF-4DGS-Ego
source venv_gb10/bin/activate
echo "PYTHON: $(which python)  ARCH: $(uname -m)"
python -c "import torch,torchvision;print('torch',torch.__version__,'tv',torchvision.__version__,'cuda',torch.cuda.is_available())"

# Node-local working dir (TMPDIR is per-job node-local, ~300+ GB).
WORK="${TMPDIR:-/tmp}/dm_hires_$SLURM_JOB_ID"
PACK="$WORK/h2o_hires"
mkdir -p "$PACK"
echo "=== node-local workdir: $WORK ===" ; df -h "$WORK" | tail -1

U="${H2O_USER}" ; P="${H2O_PASS}"
pack_subject () {
  local subj="$1" ; local url="https://h2odataset.ethz.ch/data/dataset/${subj}_ego_v1_1.tar.gz"
  echo "=== PACK $subj @ $(date) -> $PACK ==="
  curl -sL -u "$U:$P" "$url" \
    | python3 -m scripts.pack_h2o --out "$PACK" --res 224 --hand_crop_px 192
  echo "=== PACK $subj done @ $(date)  rc=$? ==="
}
# subject1 = train, subject4 = held-out test (the smallest viable cross-subject split).
pack_subject subject1
pack_subject subject4
echo "=== PACK SUMMARY ===" ; ls -la "$PACK" | head -40 ; du -sh "$PACK"
NPZ=$(ls "$PACK"/*.npz 2>/dev/null | wc -l)
S1=$(ls "$PACK"/subject1*.npz 2>/dev/null | wc -l)
S4=$(ls "$PACK"/subject4*.npz 2>/dev/null | wc -l)
echo "packed npz: total=$NPZ  subject1(train)=$S1  subject4(test)=$S4"
if [ "$S1" -eq 0 ] || [ "$S4" -eq 0 ]; then
  echo "=== FATAL: missing train or test split (S1=$S1 S4=$S4). Aborting. ==="
  exit 11
fi

# Point the config's data_root at the node-local pack.
CFG="$WORK/exp_h2o_hand_hires.yaml"
sed "s|__SET_AT_JOB_TIME__|$PACK|" configs/exp_h2o_hand_hires.yaml > "$CFG"
echo "=== data_root -> $(grep data_root $CFG) ==="

echo "=== TRAIN @ $(date) ==="
python -m scripts.train_h2o_hand --config "$CFG"
TRAIN_RC=$?
echo "=== TRAIN done @ $(date)  rc=$TRAIN_RC ==="

CKDIR=/work/scratch/dmonopoli/checkpoints/h2o_hand_hires
BEST="$CKDIR/best_cmpjpe.pt"
echo "=== saved checkpoints ===" ; ls -la "$CKDIR" 2>&1

WARM=/work/courses/3dv/team25/checkpoints/default/hand_head_final.pt
# PA-MPJPE (21-joint, Procrustes) on subject4 -- the metric the 43.6 baseline used.
if [ -f "$BEST" ]; then
  echo "=== EVAL HIRES ckpt (PA-MPJPE, 21-joint, subject4) @ $(date) ==="
  python -m scripts.eval_cmpjpe --config "$CFG" --ckpt "$BEST" \
    --h2o "$PACK" --subject subject4 --joints21 --hires --limit 400
fi
# Baseline uses the NON-hires config (pure 224px architecture, hires_hand:false)
# so the warm-start ckpt loads into the exact model that produced 43.6 -- no hires
# encoder constructed, no hand_crops fed. Same subject4 clips -> apples-to-apples.
BASE_CFG="$WORK/exp_h2o_hand_base.yaml"
sed "s|/work/scratch/dmonopoli/h2o_packed|$PACK|" configs/exp_h2o_hand.yaml > "$BASE_CFG"
echo "=== EVAL BASELINE warm-start (224px, PA-MPJPE, 21-joint, subject4) @ $(date) ==="
python -m scripts.eval_cmpjpe --config "$BASE_CFG" --ckpt "$WARM" \
  --h2o "$PACK" --subject subject4 --joints21 --limit 400

echo "=== HIRES JOB $SLURM_JOB_ID COMPLETE @ $(date) ==="
