#!/bin/bash
# World-space eval launcher + root-smoothing diagnostic (W-attack task #29).
# Stages the repo node-local (/work+/home quota-dead), overlays the updated world-space scripts
# (bundled from local into /tmp/_ws_bundle.b64), and runs the eval. With --smooth_windows it also
# reports W-MPJPE vs temporal root-smoothing window (drift-vs-bias verdict) and dumps the per-segment
# trajectories to /tmp/world_traj.pt so the smoothing sweep can be iterated offline on the Mac.
#
# Override the dataset via env: DATA_ROOT, CONFIG, MAX_SEQS, MAX_SEGS, SMOOTH, OUT, DUMP.
# Defaults run the smoothing diagnostic on HOT3D (in-domain; HOI4D world caches were cleared and
# its extrinsics are gone, so HOT3D is the available + cleaner drift-vs-bias probe).
set -uo pipefail
export XDG_CACHE_HOME=/tmp/xc TORCH_HOME=/tmp/th HF_HOME=/tmp/hf MPLCONFIGDIR=/tmp/mpl
mkdir -p /tmp/xc /tmp/th /tmp/hf /tmp/mpl
RUN=/tmp/run; rm -rf "$RUN"; mkdir -p "$RUN"
SRC=/home/dmonopoli/FF-4DGS-Ego
cp -r "$SRC/scripts" "$RUN/scripts"
ln -sfn "$SRC/diffsynth" "$RUN/diffsynth"
ln -sfn "$SRC/models"    "$RUN/models"
ln -sfn "$SRC/configs"   "$RUN/configs"
base64 -d /tmp/_ws_bundle.b64 | tar -x -C "$RUN/scripts"   # overlay updated scripts

DATA_ROOT=${DATA_ROOT:-/work/courses/3dv/team25/data/hot3d_aria/preprocessed_pinhole_f609}
CONFIG=${CONFIG:-configs/exp_p3_scalehead.yaml}
MAX_SEQS=${MAX_SEQS:-20}
MAX_SEGS=${MAX_SEGS:-2}
SMOOTH=${SMOOTH:-3,5,9,15,21,31}
OUT=${OUT:-/tmp/world_eval.json}
DUMP=${DUMP:-/tmp/world_traj.pt}

source /work/scratch/dmonopoli/venv_gb10/bin/activate
cd "$RUN"
echo "=== WORLD-SPACE EVAL + SMOOTHING DIAGNOSTIC @ $(date) on $(hostname) ==="
echo "    data_root=$DATA_ROOT  config=$CONFIG  max_seqs=$MAX_SEQS  smooth=$SMOOTH"
python -u -m scripts.eval_world_space \
  --config "$CONFIG" \
  --data_root "$DATA_ROOT" \
  --max_seqs "$MAX_SEQS" --max_segs "$MAX_SEGS" --segment_len 128 --clip_len 16 --stride 8 --wa_short 16 \
  --smooth_windows "$SMOOTH" --dump_traj "$DUMP" \
  --out "$OUT" 2>&1 \
  | grep --line-buffered -vE "Loaded GT joints|Loaded 2D GT|No calibration for|\[VIS\]"
echo "=== rc=$? ==="
echo "--- $OUT ---"; cat "$OUT" 2>/dev/null | head -60
echo "--- dump: $DUMP ---"; ls -la "$DUMP" 2>/dev/null
# To iterate offline on the Mac after copying $DUMP back:
#   python -m scripts.analyze_smoothing_diagnostic --dump world_traj.pt --windows 1,3,5,9,15,21,31
