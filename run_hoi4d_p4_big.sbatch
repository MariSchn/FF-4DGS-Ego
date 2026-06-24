#!/bin/bash
#SBATCH --account=3dv
#SBATCH --gpus=gb10:1
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --job-name=p4big
#SBATCH --output=/work/scratch/dmonopoli/joblogs/p4big_%j.out
#SBATCH --error=/work/scratch/dmonopoli/joblogs/p4big_%j.err
# ---------------------------------------------------------------------------
# HOI4D camera-frame contact-anchor A/B, BIG fire-and-forget version.
#
# Three arms, all on the de-risked pipeline (preprocess -> train -> eval ON/OFF):
#   1) smoke     : 120 steps, grad_accum 1  -> fast sanity + confirms eval works
#   2) converged : 800 steps, grad_accum 8  -> THE deliverable (effective batch 32)
#   3) ablation  : 800 steps, kp3d_abs=0    -> isolate the anchor from abs-kp supervision
#
# Arms run sequentially (QOS = 1 job). Each arm is wrapped so a failure does not
# stop later arms. Results -> /work/scratch (persistent); SLURM logs -> joblogs
# (HOME is inode-starved, see memory). MANO FK + intrinsic fixed in commit fcfd154.
#
# Submit from the LOGIN node (NOT inside an srun):  sbatch run_hoi4d_p4_big.sbatch
# If sbatch rejects --time=04:00:00 (partition cap), lower it to 02:00:00; the
# converged arm still completes (drop the ablation arm if needed).
# ---------------------------------------------------------------------------
set -uo pipefail
source /work/scratch/dmonopoli/venv_gb10/bin/activate
export XDG_CACHE_HOME=/tmp/xc TORCH_HOME=/tmp/th HF_HOME=/tmp/hf MPLCONFIGDIR=/tmp/mpl
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
mkdir -p /tmp/xc /tmp/th /tmp/hf /tmp/mpl
cd /home/dmonopoli/FF-4DGS-Ego

RESULTS=/work/scratch/dmonopoli/p4big_results
mkdir -p "$RESULTS"
echo "### HOI4D cam-anchor BIG | node $(hostname) | $(date) | commit $(git rev-parse --short HEAD) ###"

HOI4D=/work/scratch/dmonopoli/hoi4d
HANDPOSE=/work/scratch/dmonopoli/hoi4d_handpose_gt/Hand_pose/handpose_right_hand
PP=/tmp/hoi4d_pp
# Authoritative ZY20210800001 K @ 1920x1080 (camera_params/intrin.npy). Source of
# truth; --recover_k runs as a live kps2D cross-check (should agree to ~0.1%).
KVALS="1060.2955 1061.5068 971.52105 523.2619"
SEQS=(
  ZY20210800001_H1_C1_N19_S100_s02_T1
  ZY20210800001_H1_C12_N11_S169_s03_T1
  ZY20210800001_H1_C12_N26_S165_s01_T1
  ZY20210800001_H1_C13_N11_S132_s03_T1
  ZY20210800001_H1_C13_N12_S132_s03_T1
  ZY20210800001_H1_C14_N15_S158_s02_T1
  ZY20210800001_H1_C14_N17_S158_s02_T1
  ZY20210800001_H1_C2_N11_S212_s03_T2
  ZY20210800001_H1_C3_N01_S54_s05_T2
  ZY20210800001_H1_C5_N15_S57_s03_T1
  ZY20210800001_H1_C7_N11_S280_s03_T5
)

echo "######## PREPROCESS (camera-frame caches, authoritative K + kps2D cross-check) ########"
for SEQ_US in "${SEQS[@]}"; do
  SEQ_NESTED="${SEQ_US//_//}"
  echo "--- $SEQ_US ---"
  python -u -m scripts.preprocessing.preprocess_hoi4d \
    --hoi4d_root "$HOI4D" --rgb_root "$HOI4D" --handpose_root "$HANDPOSE" \
    --seq "$SEQ_NESTED" --out "$PP" --mano_model models/MANO \
    --res 224 --full_w 1920 --full_h 1080 --recover_k \
    --intrin_vals "$KVALS" 2>&1 | grep -vE "^VERIFY" || true
done

# run_arm <name> <train-override...> : train one arm, then eval anchor ON vs OFF.
run_arm () {
  local NAME="$1"; shift
  local OUTDIR="/tmp/rt_${NAME}"
  echo ""
  echo "########################## TRAIN arm=$NAME ##########################"
  python -u -m scripts.train_hand_head --config configs/exp_p4_contact_hoi4d.yaml \
    data.data_root="$PP" training.output_dir="$OUTDIR" \
    debug.enabled=true debug.max_sequences=20 "$@" 2>&1 \
    | grep --line-buffered -vE "Loaded GT joints|Loaded 2D GT|No calibration for|\[VIS\]"
  local CKPT="$OUTDIR/hand_head_final.pt"
  if [ ! -f "$CKPT" ]; then
    echo "[arm=$NAME] NO CHECKPOINT at $CKPT — skipping eval for this arm"
    return 1
  fi
  # Build anchor ON / OFF eval configs from the SAME trained checkpoint (toggle only
  # enable_root_anchor); replaces warm_start with the trained full-model dict.
  python - "$CKPT" "$NAME" <<'PY'
import sys, yaml
ckpt, name = sys.argv[1], sys.argv[2]
for tag, anchor in (("on", True), ("off", False)):
    c = yaml.safe_load(open("configs/exp_p4_contact_hoi4d.yaml"))
    c["model"]["warm_start_hand_head"] = ckpt
    c["model"]["enable_root_anchor"] = anchor
    yaml.safe_dump(c, open(f"/tmp/eval_{name}_{tag}.yaml", "w"))
print(f"[arm={name}] wrote anchor ON/OFF eval configs")
PY
  for tag in on off; do
    echo "########## EVAL arm=$NAME anchor=$tag ##########"
    python -u -m scripts.eval_hand_cam_anchor --config "/tmp/eval_${NAME}_${tag}.yaml" \
      --data_root "$PP" --max_seqs 20 --clip_len 16 --stride 8 \
      --out "$RESULTS/${NAME}_${tag}.json" 2>&1 \
      | grep -vE "Loaded GT joints|Loaded 2D GT|\[VIS\]"
    cp "$RESULTS/${NAME}_${tag}.json" "$HOME/p4big_${NAME}_${tag}.json" 2>/dev/null || true
  done
}

# ARM 1 — smoke: fast, confirms the eval step runs end to end + a first ON/OFF number.
run_arm smoke \
  training.max_steps=120 training.grad_accum_steps=1 \
  training.log_every=10 training.val_every=60 training.val_max_batches=8 || true

# ARM 2 — converged: the deliverable. Proper grad-accum (effective batch 32), 800 steps.
run_arm converged \
  training.max_steps=800 training.grad_accum_steps=8 \
  training.log_every=20 training.val_every=200 training.val_max_batches=16 || true

# ARM 3 — ablation: isolate the anchor. kp3d_abs weight -> 0 so the head has no absolute
# 3D-keypoint supervision; the anchor (via gs_depth) is the main metric-depth driver.
run_arm ablation \
  training.max_steps=800 training.grad_accum_steps=8 \
  loss_weights.kp3d_abs=0.0 \
  training.log_every=20 training.val_every=200 training.val_max_batches=16 || true

echo ""
echo "######## DONE @ $(date) ########"
echo ">>> Results in $RESULTS (and ~/p4big_*.json):"
ls -la "$RESULTS" 2>/dev/null || true
echo ">>> Raw eval JSONs (anchor ON < OFF on C-abs = anchor helps with a good gs_depth ref):"
for f in "$RESULTS"/*.json; do
  [ -f "$f" ] && { echo "----- $f -----"; cat "$f"; echo ""; }
done
