#!/bin/bash
#SBATCH --account=3dv
#SBATCH --gpus=gb10:1
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --job-name=rootdepth
#SBATCH --output=/work/scratch/dmonopoli/joblogs/rootdepth_%j.out
#SBATCH --error=/work/scratch/dmonopoli/joblogs/rootdepth_%j.err
# ---------------------------------------------------------------------------
# Root-depth supervision experiment (Cyrus #31): "add a scale estimator only for
# the hand root (translation) and supervise it with larger weights."
#
# Scoping finding: the up-weighted root-DEPTH supervision Cyrus asked for ALREADY
# EXISTS as a wired-but-unused knob — loss_weights.transl_z_weight scales axis-2
# (camera depth) of the direct transl loss (train_hand_head.py:1286 ->
# hamer_losses.py:205, "up-weight axis 2 (camera depth) to attack the hand-root
# absolute-placement error the W-MPJPE diagnostic localised"). It defaults to 1.0.
# So the first, ZERO-NEW-CODE test of #31 is to turn it up and measure C-abs. This
# isolates the HEAD's own absolute placement: anchor OFF (the contact anchor came
# back neutral), eval --contact_gate off, kp3d_abs as the existing absolute lever.
#
#   arm             transl_z_weight  kp3d_abs   question
#   control         1.0              0.3        baseline head (≈ contact-run control 105.8)
#   rootz3          3.0              0.3        up-weight root DEPTH supervision 3x
#   rootz3_absbump  3.0              0.5        + stronger absolute-keypoint supervision
#
# DECISION: rootz3 C-abs < control => up-weighting the root-depth supervision
# improves absolute placement (Cyrus #31 validated config-only). A dedicated
# root-depth HEAD module is the follow-up build ONLY if this simple knob plateaus.
#
# Same stable operating point as the contact run (120 steps, grad_accum 1).
# Submit from the LOGIN node:  sbatch run_hoi4d_rootdepth_gb10.sbatch
# ---------------------------------------------------------------------------
set -uo pipefail
source /work/scratch/dmonopoli/venv_gb10/bin/activate
export XDG_CACHE_HOME=/tmp/xc TORCH_HOME=/tmp/th HF_HOME=/tmp/hf MPLCONFIGDIR=/tmp/mpl
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
mkdir -p /tmp/xc /tmp/th /tmp/hf /tmp/mpl
cd /home/dmonopoli/FF-4DGS-Ego

RESULTS=/work/scratch/dmonopoli/rootdepth_results
mkdir -p "$RESULTS"
echo "### ROOT-DEPTH (#31) | node $(hostname) | $(date) | commit $(git rev-parse --short HEAD) ###"

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

# run_arm <name> <transl_z_weight> <kp3d_abs>: train at the stable operating point with
# the anchor OFF (pure head), then eval C-abs with the anchor disabled (--contact_gate off).
run_arm () {
  local NAME="$1"; local TZW="$2"; local KABS="$3"
  local OUTDIR="/tmp/rt_${NAME}"
  echo ""
  echo "########################## TRAIN arm=$NAME transl_z=$TZW kp3d_abs=$KABS ##########################"
  python -u -m scripts.train_hand_head --config configs/exp_p4_contact_hoi4d.yaml \
    data.data_root="$PP" training.output_dir="$OUTDIR" \
    training.max_steps=120 training.grad_accum_steps=1 \
    training.val_every=60 training.log_every=20 \
    model.enable_root_anchor=false loss_weights.root_anchor=0.0 \
    loss_weights.transl_z_weight="$TZW" loss_weights.kp3d_abs="$KABS" \
    training.use_contact_gate=false \
    debug.enabled=true debug.max_sequences=20 2>&1 \
    | grep --line-buffered -vE "Loaded GT joints|Loaded 2D GT|No calibration for|\[VIS\]"
  local CKPT="$OUTDIR/hand_head_final.pt"
  if [ ! -f "$CKPT" ]; then echo "[arm=$NAME] NO CHECKPOINT at $CKPT — skipping eval"; return 1; fi
  python - "$CKPT" "$NAME" <<'PY'
import sys, yaml
ckpt, name = sys.argv[1], sys.argv[2]
c = yaml.safe_load(open("configs/exp_p4_contact_hoi4d.yaml"))
c["model"]["warm_start_hand_head"] = ckpt
c["model"]["enable_root_anchor"] = False     # pure head placement; eval gate 'off' also disables it
yaml.safe_dump(c, open(f"/tmp/eval_{name}.yaml", "w"))
print(f"[arm={name}] wrote eval config")
PY
  echo "########## EVAL arm=$NAME (anchor off) ##########"
  python -u -m scripts.eval_hand_cam_anchor --config "/tmp/eval_${NAME}.yaml" \
    --data_root "$PP" --max_seqs 20 --clip_len 16 --stride 8 \
    --contact_gate off --out "$RESULTS/${NAME}.json" 2>&1 \
    | grep -vE "Loaded GT joints|Loaded 2D GT|\[VIS\]"
  cp "$RESULTS/${NAME}.json" "$HOME/rootdepth_${NAME}.json" 2>/dev/null || true
}

run_arm control        1.0 0.3 || true
run_arm rootz3         3.0 0.3 || true
run_arm rootz3_absbump 3.0 0.5 || true

echo ""
echo "######## DONE @ $(date) ########"
echo ">>> C-abs per arm (lower better). rootz3 < control => up-weighting root-depth"
echo ">>> supervision improves absolute placement (Cyrus #31, config-only):"
for f in "$RESULTS"/control.json "$RESULTS"/rootz3.json "$RESULTS"/rootz3_absbump.json; do
  [ -f "$f" ] && { echo "----- $f -----"; cat "$f"; echo ""; }
done
