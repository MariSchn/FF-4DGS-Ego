#!/bin/bash
# One-shot HOI4D camera-frame contact-anchor A/B (no extrinsics):
#   preprocess (extrinsics-optional) -> train contact head on HOI4D with the
#   validated-depth backbone (best_depth.pt) -> eval C-abs anchor ON vs OFF.
# This is the FAIR test the HOT3D run could not do: HOT3D gs_depth was a worse
# reference than the head (Δgs ~122 vs C-abs ~50); HOI4D gs_depth is validated.
#
# Intrinsics: NOT on this mirror (camera/recon/split_0 has only info.json). But every
# handpose pickle carries kps2D[21,2], so preprocess --recover_k solves the full-res K
# by least-squares from the MANO 3D joints vs kps2D (camera is fixed across HOI4D).
#
# Run on a gb10 node (inside tmux):
#   srun --account=3dv --gpus=gb10:1 --mem=48G --time=02:00:00 --pty bash run_hoi4d_cam_anchor_gb10.sh
set -uo pipefail
# GUARD: venv_gb10 is aarch64 (built for the gb10 nodes). On the x86 LOGIN node its
# torch .so files can't load (libtorch_global_deps.so OSError). Refuse in 1s instead of
# failing slowly mid-run. If this trips, you forgot to srun onto a gb10 node first.
if [ "$(uname -m)" != "aarch64" ]; then
  echo "ERROR: arch=$(uname -m) (expected aarch64). You are on the x86 LOGIN node, not a"
  echo "       gb10 compute node. Start an interactive gb10 shell first, THEN re-run:"
  echo "         srun --account=3dv --gpus=gb10:1 --mem=48G --time=02:30:00 --pty bash"
  echo "       (wait for the studgpu-spark... prompt before running this script)"
  exit 3
fi
source /work/scratch/dmonopoli/venv_gb10/bin/activate
export XDG_CACHE_HOME=/tmp/xc TORCH_HOME=/tmp/th HF_HOME=/tmp/hf MPLCONFIGDIR=/tmp/mpl
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
mkdir -p /tmp/xc /tmp/th /tmp/hf /tmp/mpl
cd /home/dmonopoli/FF-4DGS-Ego
OUT=/tmp/p4out; mkdir -p "$OUT"; LOG="$OUT/hoi4d_run.log"
exec > >(tee "$LOG") 2>&1
echo "### HOI4D cam-anchor A/B | node $(hostname) | $(date) ###"

HOI4D=/work/scratch/dmonopoli/hoi4d
HANDPOSE=/work/scratch/dmonopoli/hoi4d_handpose_gt/Hand_pose/handpose_right_hand
PP=/tmp/hoi4d_pp
# The 11 flat seq dirs that have BOTH images/ AND right-hand handpose pickles
# (verified on-cluster). Handpose path derives by replacing _ with /.
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

echo "######## PREPROCESS (camera-frame caches, K recovered from kps2D) ########"
for SEQ_US in "${SEQS[@]}"; do
  SEQ_NESTED="${SEQ_US//_//}"
  echo "--- $SEQ_US ---"
  python -u -m scripts.preprocessing.preprocess_hoi4d \
    --hoi4d_root "$HOI4D" --rgb_root "$HOI4D" --handpose_root "$HANDPOSE" \
    --seq "$SEQ_NESTED" --out "$PP" --mano_model models/MANO \
    --res 224 --full_w 1920 --full_h 1080 --recover_k 2>&1 \
    | grep -vE "^VERIFY" || true
done

echo "######## TRAIN contact head on HOI4D (anchor lr=1e-3, 120 steps) ########"
python -u -m scripts.train_hand_head --config configs/exp_p4_contact_hoi4d.yaml \
  data.data_root="$PP" training.output_dir=/tmp/rt_hoi4d \
  training.max_steps=120 training.grad_accum_steps=1 \
  training.log_every=10 training.val_every=60 training.val_max_batches=8 \
  debug.enabled=true debug.max_sequences=20 2>&1 \
  | grep --line-buffered -vE "Loaded GT joints|Loaded 2D GT|No calibration for|\[VIS\]"

echo "######## build eval configs (anchor ON / OFF, same trained ckpt) ########"
python - <<'PY'
import yaml
base = yaml.safe_load(open("configs/exp_p4_contact_hoi4d.yaml"))
base["model"]["warm_start_hand_head"] = "/tmp/rt_hoi4d/hand_head_final.pt"
yaml.safe_dump(base, open("/tmp/eval_hoi4d_on.yaml", "w"))
off = yaml.safe_load(open("configs/exp_p4_contact_hoi4d.yaml"))
off["model"]["warm_start_hand_head"] = "/tmp/rt_hoi4d/hand_head_final.pt"
off["model"]["enable_root_anchor"] = False
yaml.safe_dump(off, open("/tmp/eval_hoi4d_off.yaml", "w"))
print("wrote /tmp/eval_hoi4d_on.yaml + /tmp/eval_hoi4d_off.yaml")
PY

echo "######## EVAL C-abs: anchor ON ########"
python -u -m scripts.eval_hand_cam_anchor --config /tmp/eval_hoi4d_on.yaml \
  --data_root "$PP" --max_seqs 20 --clip_len 16 --stride 8 --out "$OUT/hoi4d_cam_on.json" 2>&1 \
  | grep -vE "Loaded GT joints|Loaded 2D GT|\[VIS\]"
cp "$OUT/hoi4d_cam_on.json" "$HOME/hoi4d_cam_on.json" 2>/dev/null || true

echo "######## EVAL C-abs: anchor OFF ########"
python -u -m scripts.eval_hand_cam_anchor --config /tmp/eval_hoi4d_off.yaml \
  --data_root "$PP" --max_seqs 20 --clip_len 16 --stride 8 --out "$OUT/hoi4d_cam_off.json" 2>&1 \
  | grep -vE "Loaded GT joints|Loaded 2D GT|\[VIS\]"
cp "$OUT/hoi4d_cam_off.json" "$HOME/hoi4d_cam_off.json" 2>/dev/null || true

echo "######## DONE @ $(date) ########"
echo ">>> Compare the two 'OURS HOI4D camera-frame' C-abs lines: if ANCHOR ON < anchor OFF,"
echo ">>> the contact anchor improves absolute placement when gs_depth is a good reference."
