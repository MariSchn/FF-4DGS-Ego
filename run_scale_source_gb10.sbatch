#!/bin/bash
#SBATCH --account=3dv
#SBATCH --gpus=gb10:1
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --job-name=scalesrc
#SBATCH --output=/work/scratch/dmonopoli/joblogs/scalesrc_%j.out
#SBATCH --error=/work/scratch/dmonopoli/joblogs/scalesrc_%j.err
# ---------------------------------------------------------------------------
# Scale-source ablation — THE thesis test (issue #1).
#
# Q: which metric-scale source best places the FROZEN up-to-scale gs_depth onto
#    HOI4D's DENSE GT sensor depth, scored ONLY on non-hand pixels (non-circular)?
#      none(s=1) | HAND (in-scene MANO, ours) | oracle(median gt/gs) | fm(UniDepth, opt)
#    Claim:  HAND approx oracle  AND  HAND < fm  =>  the hand drives the metric coupling.
#
# Pipeline (QOS=1, sequential):
#   0) preprocess the 11 ZY20210800001 handpose seqs -> /tmp/hoi4d_pp (hand caches + K).
#   1) LAYOUT AUDIT: per seq, does <raw>/<seq_us>/{images,raw_depth} exist? (self-diagnose:
#      eval scores only seqs present in BOTH --raw raw_depth AND --pp hand caches).
#   2) SMOKE: eval_scale_source --smoke (1 seq, 2 clips) — confirms gs_depth fwd + scoring,
#      prints the BOTH-present intersection count. If it says "0 seqs", the depth for these
#      seqs isn't staged under --raw (only images were) -> stop and stage raw_depth.
#   3) FULL: eval_scale_source over every BOTH-present seq -> the deliverable JSON.
#
# fm(UniDepth) auto-skips on aarch64 (graceful try-import); run that source on the x86
# venv_unidepth stage if we want the FM column. RGB/K source-of-truth = p4big commit.
#
# Submit from the LOGIN node:  sbatch run_scale_source_gb10.sbatch
# ---------------------------------------------------------------------------
set -uo pipefail
source /work/scratch/dmonopoli/venv_gb10/bin/activate
export XDG_CACHE_HOME=/tmp/xc TORCH_HOME=/tmp/th HF_HOME=/tmp/hf MPLCONFIGDIR=/tmp/mpl
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
mkdir -p /tmp/xc /tmp/th /tmp/hf /tmp/mpl
cd /home/dmonopoli/FF-4DGS-Ego

RESULTS=/work/scratch/dmonopoli/scalesrc_results
mkdir -p "$RESULTS"
echo "### SCALE-SOURCE ablation | node $(hostname) | $(date) | commit $(git rev-parse --short HEAD) ###"

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

echo "######## PREPROCESS (hand caches + cam_intrinsics, authoritative K) ########"
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
echo "######## LAYOUT AUDIT (the one real unknown: is raw_depth staged for these seqs?) ########"
BOTH=0
for SEQ_US in "${SEQS[@]}"; do
  HAS_IMG="no"; [ -d "$HOI4D/$SEQ_US/images" ] && HAS_IMG="yes"
  HAS_DEP="no"; [ -d "$HOI4D/$SEQ_US/raw_depth" ] && HAS_DEP="yes"
  HAS_CACHE="no"; [ -f "$PP/$SEQ_US/hand_data/gt_joints_cache_cam_v2.pt" ] && HAS_CACHE="yes"
  if [ "$HAS_DEP" = "yes" ] && [ "$HAS_CACHE" = "yes" ]; then BOTH=$((BOTH+1)); fi
  printf "  %-44s images=%s raw_depth=%s handcache=%s\n" "$SEQ_US" "$HAS_IMG" "$HAS_DEP" "$HAS_CACHE"
done
echo ">>> seqs with BOTH raw_depth AND hand cache (the scorable set): $BOTH / ${#SEQS[@]}"
if [ "$BOTH" -eq 0 ]; then
  echo "!!! 0 scorable seqs: raw_depth for the handpose seqs is NOT staged under $HOI4D."
  echo "!!! These seqs only have images/ (p4big never needed depth). Stage raw_depth/ per"
  echo "!!! underscored seq, then re-run. Skipping eval (it would score nothing)."
  exit 3
fi

echo ""
echo "######## SMOKE (1 seq, 2 clips) ########"
python -u -m scripts.eval_scale_source --config configs/exp_hoi4d_depth.yaml \
  --pp "$PP" --raw "$HOI4D" --smoke --out "$RESULTS/scale_source_smoke.json" 2>&1 \
  | grep -vE "Loaded GT joints|Loaded 2D GT|\[VIS\]" || true
cp "$RESULTS/scale_source_smoke.json" "$HOME/scalesrc_smoke.json" 2>/dev/null || true

echo ""
echo "######## FULL (all scorable seqs) ########"
python -u -m scripts.eval_scale_source --config configs/exp_hoi4d_depth.yaml \
  --pp "$PP" --raw "$HOI4D" --max_clips_per_seq 8 --out "$RESULTS/scale_source.json" 2>&1 \
  | grep -vE "Loaded GT joints|Loaded 2D GT|\[VIS\]" || true
cp "$RESULTS/scale_source.json" "$HOME/scalesrc.json" 2>/dev/null || true

echo ""
echo "######## DONE @ $(date) ########"
echo ">>> Deliverable: $RESULTS/scale_source.json (and ~/scalesrc.json)"
for f in "$RESULTS"/scale_source*.json; do
  [ -f "$f" ] && { echo "----- $f -----"; cat "$f"; echo ""; }
done
