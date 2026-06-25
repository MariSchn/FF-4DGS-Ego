#!/bin/bash
#SBATCH --account=3dv
#SBATCH --gpus=gb10:1
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --job-name=wabs
#SBATCH --output=/work/scratch/dmonopoli/joblogs/wabs_%j.out
#SBATCH --error=/work/scratch/dmonopoli/joblogs/wabs_%j.err
# ---------------------------------------------------------------------------
# Self-contained kp3d_abs (absolute hand-depth) retrain + world-space eval, HOT3D.
#
# Fixes the hand ROOT ABSOLUTE DEPTH (z_hand) by turning on kp3d_abs (0 -> 0.3) on a
# frozen-backbone hand-head-only fine-tune. z_hand drives BOTH C_abs (driver #2) AND
# the derived scene scale s_hand = median(z_hand/scene_depth) (driver #3), so one
# retrain attacks both W-MPJPE drivers. Proven probe: W 308 -> 250, C_abs 115 -> 53.
#
# Robust by construction:
#   - bash shebang so `source` works (sbatch --wrap defaults to /bin/sh = dash; that
#     broke job 101490 with "source: not found / python: not found").
#   - checkpoint -> node-local /tmp (860G), NOT /work (a /work save produced a 0-byte
#     best_val_loss.pt). Only the small eval JSON is persisted; the multi-GB head is
#     copied to team25 best-effort but the result does not depend on it.
#   - train and eval in ONE job, so the /tmp checkpoint is consumed before the node is
#     released (no persistent multi-GB storage needed).
#
# Submit from the LOGIN node:  sbatch run_wabs_gb10.sbatch
# ---------------------------------------------------------------------------
set -uo pipefail
source /work/scratch/dmonopoli/venv_gb10/bin/activate
export XDG_CACHE_HOME=/tmp/xc TORCH_HOME=/tmp/th HF_HOME=/tmp/hf MPLCONFIGDIR=/tmp/mpl
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
mkdir -p /tmp/xc /tmp/th /tmp/hf /tmp/mpl
cd /home/dmonopoli/FF-4DGS-Ego
echo "### WABS retrain+eval | node $(hostname) | $(date) | commit $(git rev-parse --short HEAD) ###"

HOT3D=/work/courses/3dv/team25/data/hot3d_aria/preprocessed_pinhole_f609
OUT=/tmp/rt_wabs
RES=/work/scratch/dmonopoli/wabs_results
mkdir -p "$RES"

echo "######## TRAIN exp_p3_wabs (kp3d_abs 0->0.3, frozen backbone, GS off) ########"
python -u -m scripts.train_hand_head --config configs/exp_p3_wabs.yaml \
  data.data_root="$HOT3D" training.output_dir="$OUT" \
  training.max_steps=600 training.val_every=100 training.log_every=20 2>&1 \
  | grep --line-buffered -vE "Loaded GT joints|Loaded 2D GT|No calibration for|\[VIS\]"

CKPT="$OUT/hand_head_final.pt"
if [ ! -f "$CKPT" ]; then
  echo "[wabs] NO CHECKPOINT at $CKPT — training did not finish; aborting eval"
  exit 1
fi
# Best-effort persist (not required for the result; eval uses the /tmp copy below).
cp "$CKPT" /work/courses/3dv/team25/checkpoints/p3_wabs/hand_head_final.pt 2>/dev/null \
  && echo "[wabs] persisted head to team25" \
  || echo "[wabs] could not persist head to team25 (quota?) — fine, using /tmp copy"

echo "######## build eval config (GS ON, warm_start = trained head) ########"
python - "$CKPT" <<'PY'
import sys, yaml
ckpt = sys.argv[1]
c = yaml.safe_load(open("configs/exp_p3_wabs_eval.yaml"))
c["model"]["warm_start_hand_head"] = ckpt
yaml.safe_dump(c, open("/tmp/wabs_eval.yaml", "w"))
print("wrote /tmp/wabs_eval.yaml warm_start ->", ckpt)
PY

echo "######## EVAL world-space (HOT3D, robust scale) ########"
python -u -m scripts.eval_world_space --config /tmp/wabs_eval.yaml \
  --data_root "$HOT3D" --max_seqs 3 --max_segs 2 \
  --segment_len 128 --clip_len 16 --stride 8 --wa_short 16 \
  --robust_scale --out "$RES/wabs_w.json" 2>&1 \
  | grep -vE "Loaded GT joints|Loaded 2D GT|\[VIS\]"
cp "$RES/wabs_w.json" "$HOME/wabs_w.json" 2>/dev/null || true

echo "######## DONE @ $(date) ########"
echo ">>> Pre-retrain baseline on these segs: W~290, C_abs~179, s_hand/s_gt~1.35."
echo ">>> Success = W drops toward reanchor16 (~64) and C_abs -> ~120, scale ratio -> 1."
echo ">>> wabs_w.json aggregate:"
cat "$RES/wabs_w.json" 2>/dev/null || true
