#!/bin/bash
# (a) Predicted metric depth vs dense GT depth on HOI4D raw_depth (Cyrus check a).
# Node-local /tmp staging (shared FS quota-dead). Overlays the two new scripts, auto-finds the
# HOI4D-depth-trained checkpoint, and runs the depth-quality eval. Override via env: CONFIG,
# DATA_ROOT, CKPT, MAX_SEQS, NUM_CLIPS.
set -uo pipefail
export XDG_CACHE_HOME=/tmp/xc TORCH_HOME=/tmp/th HF_HOME=/tmp/hf MPLCONFIGDIR=/tmp/mpl
mkdir -p /tmp/xc /tmp/th /tmp/hf /tmp/mpl
RUN=/tmp/run; rm -rf "$RUN"; mkdir -p "$RUN"
SRC=/home/dmonopoli/FF-4DGS-Ego
cp -r "$SRC/scripts" "$RUN/scripts"
ln -sfn "$SRC/diffsynth" "$RUN/diffsynth"
ln -sfn "$SRC/models"    "$RUN/models"
ln -sfn "$SRC/configs"   "$RUN/configs"
base64 -d /tmp/_depth_bundle.b64 | tar -x -C "$RUN/scripts"   # overlay new scripts

CONFIG=${CONFIG:-configs/exp_hoi4d_depth.yaml}
DATA_ROOT=${DATA_ROOT:-/work/scratch/dmonopoli/hoi4d}
MAX_SEQS=${MAX_SEQS:-12}     # span more HOI4D object classes so the figure scenes are diverse
NUM_CLIPS=${NUM_CLIPS:-40}
FIG_N=${FIG_N:-4}
# auto-find the most recent trained checkpoint (best_* preferred, else newest .pt)
CKPT=${CKPT:-}
if [ -z "$CKPT" ]; then
  CKPT=$(ls -t /work/scratch/dmonopoli/checkpoints/hoi4d_depth/best*.pt 2>/dev/null | head -1)
  [ -z "$CKPT" ] && CKPT=$(ls -t /work/scratch/dmonopoli/checkpoints/hoi4d_depth/*.pt 2>/dev/null | head -1)
fi

source /work/scratch/dmonopoli/venv_gb10/bin/activate
cd "$RUN"
echo "=== PRED-DEPTH vs GT (HOI4D dense) @ $(date) on $(hostname) ==="
echo "    config=$CONFIG  data_root=$DATA_ROOT  ckpt=${CKPT:-<none: base only>}  max_seqs=$MAX_SEQS"
echo "--- checkpoints present in hoi4d_depth dir ---"; ls -t /work/scratch/dmonopoli/checkpoints/hoi4d_depth/*.pt 2>/dev/null | head
CKARG=""; [ -n "$CKPT" ] && CKARG="--checkpoint $CKPT"
rm -f /tmp/depth_fig.npz
python -u -m scripts.eval_pred_depth_vs_gt \
  --config "$CONFIG" --data_root "$DATA_ROOT" $CKARG \
  --max_seqs "$MAX_SEQS" --num_clips "$NUM_CLIPS" --fig_npz /tmp/depth_fig.npz --fig_n "$FIG_N" 2>&1 \
  | grep --line-buffered -vE "Loaded GT joints|No calibration for|\[VIS\]"
echo "=== rc=$? ==="
# Stream the compact raw-array npz back through stdout (shared FS quota-dead; node /tmp ephemeral).
# The Mac-side tee captures it; the parent decodes and renders the polished figure locally.
echo "=== FIG NPZ (base64, single line between markers) ==="
if [ -f /tmp/depth_fig.npz ]; then
  echo "===NPZ depth_fig.npz ($(wc -c < /tmp/depth_fig.npz) bytes)==="
  base64 -w0 /tmp/depth_fig.npz; echo
  echo "===NPZEND==="
fi
