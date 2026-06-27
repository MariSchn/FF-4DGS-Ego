#!/bin/bash
#SBATCH --account=3dv
#SBATCH --gpus=gb10:1
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --job-name=contactdec
#SBATCH --output=/work/scratch/dmonopoli/joblogs/contactdec_%j.out
#SBATCH --error=/work/scratch/dmonopoli/joblogs/contactdec_%j.err
# ---------------------------------------------------------------------------
# Contact Phase 2 — DECISION-ONLY fallback. Runs just the two CONTACT arms
# (contact_only, contact_kp) so they get the full time budget if the 4-arm
# run_hoi4d_contact_phase2_gb10.sbatch hit the 3h wall before contact_only's
# eval saved. control.json + proxy_only.json from that run are reused (already
# on /work). Only 2 arms -> fits comfortably even at the slow ~40s/it pace.
# Decision: contact_only C-abs < control(105.8) => anchor helps alone;
#           contact_only < proxy_only         => contact gating beats the band.
# Submit from the LOGIN node:  sbatch run_hoi4d_contact_decision_gb10.sbatch
# ---------------------------------------------------------------------------
set -uo pipefail
source /work/scratch/dmonopoli/venv_gb10/bin/activate
export XDG_CACHE_HOME=/tmp/xc TORCH_HOME=/tmp/th HF_HOME=/tmp/hf MPLCONFIGDIR=/tmp/mpl
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
mkdir -p /tmp/xc /tmp/th /tmp/hf /tmp/mpl
cd /home/dmonopoli/FF-4DGS-Ego

RESULTS=/work/scratch/dmonopoli/contact_p2_results
mkdir -p "$RESULTS"
echo "### CONTACT PHASE 2 DECISION-ONLY | node $(hostname) | $(date) | commit $(git rev-parse --short HEAD) ###"

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

echo "######## PREPROCESS ########"
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
echo "######## BUILD CONTACT CACHE (5cm) ########"
python -u -m scripts.build_contact_cache --pp "$PP" --raw "$HOI4D" --res 224 --thresh_m 0.05 || true

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
  if [ ! -f "$CKPT" ]; then echo "[arm=$NAME] NO CHECKPOINT — skipping eval"; return 1; fi
  python - "$CKPT" "$NAME" <<'PY'
import sys, yaml
ckpt, name = sys.argv[1], sys.argv[2]
c = yaml.safe_load(open("configs/exp_p4_contact_hoi4d.yaml"))
c["model"]["warm_start_hand_head"] = ckpt
c["model"]["enable_root_anchor"] = True
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

run_arm contact_only oracle model.enable_root_anchor=true loss_weights.kp3d_abs=0.0 training.use_contact_gate=true || true
run_arm contact_kp   oracle model.enable_root_anchor=true loss_weights.kp3d_abs=0.3 training.use_contact_gate=true || true

echo ""
echo "######## DONE @ $(date) ########"
echo ">>> decision: contact_only C-abs < control(105.8) => anchor helps alone; < proxy_only => contact beats band"
for f in "$RESULTS"/control.json "$RESULTS"/proxy_only.json "$RESULTS"/contact_only.json "$RESULTS"/contact_kp.json; do
  [ -f "$f" ] && { echo "----- $f -----"; cat "$f"; echo ""; }
done
