#!/bin/bash
#SBATCH --account=3dv
#SBATCH --gpus=gb10:1
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --job-name=contactp2
#SBATCH --output=/work/scratch/dmonopoli/joblogs/contactp2_%j.out
#SBATCH --error=/work/scratch/dmonopoli/joblogs/contactp2_%j.err
# ---------------------------------------------------------------------------
# Contact Anchor PHASE 2 — isolation experiment (the salvage).
#
# The scale-source contact-stratified test (job 101529) showed the hand recovers
# the scene's metric scale almost perfectly AT contact (s_contact=1.005 vs oracle
# 1.022) but biased in free space (0.804). The old anchor gate band_m=0.5 (50cm)
# fires across BOTH -> it pulls toward biased free-space depth. Phase 2 gates the
# anchor by an explicit GT contact signal (|wrist_z_GT - GT_dense_depth| < 5cm).
#
# Pipeline (QOS=1, sequential): preprocess 11 seqs -> build contact_cache -> 4 arms
# at the STABLE operating point (120 steps, grad_accum 1; the 800-step p4big arms
# diverged). Each arm = train then eval C-abs. kp3d_abs=0 isolates the anchor from
# absolute-keypoint supervision.
#
#   arm           train gate         kp3d_abs   eval gate     question
#   control       (no anchor)        0.0        off           head with no metric signal
#   proxy_only    band 0.5 (proxy)   0.0        proxy         the OLD wide-band anchor
#   contact_only  GT contact         0.0        oracle        the Phase-2 anchor, ISOLATED
#   contact_kp    GT contact         0.3        oracle        Phase-2 anchor + abs-kp (best)
#
# DECISION: contact_only C-abs < control  => the contact anchor adds metric placement
# on its own; contact_only < proxy_only   => contact gating beats the wide band.
# The oracle eval gate uses GT contact -> report contact_only/contact_kp as an
# oracle-gate CEILING; proxy_only is the deployable (no-GT) number.
#
# Submit from the LOGIN node:  sbatch run_hoi4d_contact_phase2_gb10.sbatch
# ---------------------------------------------------------------------------
set -uo pipefail
source /work/scratch/dmonopoli/venv_gb10/bin/activate
export XDG_CACHE_HOME=/tmp/xc TORCH_HOME=/tmp/th HF_HOME=/tmp/hf MPLCONFIGDIR=/tmp/mpl
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
mkdir -p /tmp/xc /tmp/th /tmp/hf /tmp/mpl
cd /home/dmonopoli/FF-4DGS-Ego

RESULTS=/work/scratch/dmonopoli/contact_p2_results
mkdir -p "$RESULTS"
echo "### CONTACT PHASE 2 | node $(hostname) | $(date) | commit $(git rev-parse --short HEAD) ###"

HOI4D=/work/scratch/dmonopoli/hoi4d
HANDPOSE=/work/scratch/dmonopoli/hoi4d_handpose_gt/Hand_pose/handpose_right_hand
PP=/tmp/hoi4d_pp
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

echo "######## PREPROCESS (hand caches + cam_intrinsics) ########"
for SEQ_US in "${SEQS[@]}"; do
  SEQ_NESTED="${SEQ_US//_//}"
  echo "--- $SEQ_US ---"
  python -u -m scripts.preprocessing.preprocess_hoi4d \
    --hoi4d_root "$HOI4D" --rgb_root "$HOI4D" --handpose_root "$HANDPOSE" \
    --seq "$SEQ_NESTED" --out "$PP" --mano_model models/MANO \
    --res 224 --full_w 1920 --full_h 1080 --recover_k \
    --intrin_vals "$KVALS" 2>&1 | grep -vE "^VERIFY" || true
done

echo ""
echo "######## BUILD CONTACT CACHE (GT wrist on GT surface, 5cm) ########"
python -u -m scripts.build_contact_cache --pp "$PP" --raw "$HOI4D" --res 224 --thresh_m 0.05 || true

# run_arm <name> <eval-gate> <train-overrides...>: train at the stable operating point,
# then eval C-abs with the arm's gate (the trained full-model dict replaces warm_start).
run_arm () {
  local NAME="$1"; local GATE="$2"; shift 2
  local OUTDIR="/tmp/rt_${NAME}"
  echo ""
  echo "########################## TRAIN arm=$NAME gate=$GATE ##########################"
  python -u -m scripts.train_hand_head --config configs/exp_p4_contact_hoi4d.yaml \
    data.data_root="$PP" training.output_dir="$OUTDIR" \
    training.max_steps=120 training.grad_accum_steps=1 \
    training.val_every=60 training.log_every=20 \
    debug.enabled=true debug.max_sequences=20 "$@" 2>&1 \
    | grep --line-buffered -vE "Loaded GT joints|Loaded 2D GT|No calibration for|\[VIS\]"
  local CKPT="$OUTDIR/hand_head_final.pt"
  if [ ! -f "$CKPT" ]; then echo "[arm=$NAME] NO CHECKPOINT at $CKPT — skipping eval"; return 1; fi
  python - "$CKPT" "$NAME" <<'PY'
import sys, yaml
ckpt, name = sys.argv[1], sys.argv[2]
c = yaml.safe_load(open("configs/exp_p4_contact_hoi4d.yaml"))
c["model"]["warm_start_hand_head"] = ckpt
c["model"]["enable_root_anchor"] = True          # eval flag --contact_gate off disables it for 'control'
yaml.safe_dump(c, open(f"/tmp/eval_{name}.yaml", "w"))
print(f"[arm={name}] wrote eval config")
PY
  echo "########## EVAL arm=$NAME gate=$GATE ##########"
  python -u -m scripts.eval_hand_cam_anchor --config "/tmp/eval_${NAME}.yaml" \
    --data_root "$PP" --max_seqs 20 --clip_len 16 --stride 8 \
    --contact_gate "$GATE" --out "$RESULTS/${NAME}.json" 2>&1 \
    | grep -vE "Loaded GT joints|Loaded 2D GT|\[VIS\]"
  cp "$RESULTS/${NAME}.json" "$HOME/contact_p2_${NAME}.json" 2>/dev/null || true
}

run_arm control      off    model.enable_root_anchor=false loss_weights.kp3d_abs=0.0 training.use_contact_gate=false || true
run_arm proxy_only   proxy  model.enable_root_anchor=true  loss_weights.kp3d_abs=0.0 training.use_contact_gate=false || true
run_arm contact_only oracle model.enable_root_anchor=true  loss_weights.kp3d_abs=0.0 training.use_contact_gate=true  || true
run_arm contact_kp   oracle model.enable_root_anchor=true  loss_weights.kp3d_abs=0.3 training.use_contact_gate=true  || true

echo ""
echo "######## DONE @ $(date) ########"
echo ">>> C-abs per arm (lower better). contact_only < control => anchor helps alone;"
echo ">>> contact_only < proxy_only => contact gating beats the wide band:"
for f in "$RESULTS"/control.json "$RESULTS"/proxy_only.json "$RESULTS"/contact_only.json "$RESULTS"/contact_kp.json; do
  [ -f "$f" ] && { echo "----- $f -----"; cat "$f"; echo ""; }
done
