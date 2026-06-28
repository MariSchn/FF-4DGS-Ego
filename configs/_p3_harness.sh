#!/bin/bash
# Self-contained P3 verdict harness. Beats the cluster walls:
#  - /work + /home are quota-dead -> stage everything node-local in /tmp
#  - residuals are wandb-only (never printed) -> patch a /tmp copy of the train
#    script to print scale_residual_m / obj_depth_residual_m / hand_depth_residual_m
#  - I/O + gsplat make full runs too slow to validate -> few sequences (8) so a
#    validation (with the residual verdict) lands in ~6-8 min.
# Usage:  bash _p3_harness.sh <config.yaml> [extra key=val overrides...]
set -uo pipefail
CFG="${1:?config required}"; shift || true
export XDG_CACHE_HOME=/tmp/xc TORCH_HOME=/tmp/th HF_HOME=/tmp/hf MPLCONFIGDIR=/tmp/mpl
mkdir -p /tmp/xc /tmp/th /tmp/hf /tmp/mpl
RUN=/tmp/run
rm -rf "$RUN"; mkdir -p "$RUN"
SRC=/home/dmonopoli/FF-4DGS-Ego
cp -r "$SRC/scripts" "$RUN/scripts"
ln -sfn "$SRC/diffsynth" "$RUN/diffsynth"
ln -sfn "$SRC/models"    "$RUN/models"
ln -sfn "$SRC/configs"   "$RUN/configs"
source /work/scratch/dmonopoli/venv_gb10/bin/activate
# --- patch: print the residual verdict right after each validation ---
python - <<'PY'
f="/tmp/run/scripts/train_hand_head.py"
s=open(f).read()
anchor="= run_validation("
idx=s.index(anchor)
eol=s.index(chr(10), idx)
inject=(chr(10) + " "*20 +
        "print('  >> RESID scale=%.2fcm obj=%.2fcm hand_depth=%.2fcm' % "
        "(val_terms['scale_residual_m']*100, val_terms['obj_depth_residual_m']*100, "
        "val_terms['hand_depth_residual_m']*100), flush=True)")
open(f,"w").write(s[:eol]+inject+s[eol:])
print("PATCHED residual-print at offset", idx, flush=True)
PY
cd "$RUN"
echo "=== RUN $CFG @ $(date) on $(hostname) ==="
python -u -m scripts.train_hand_head --config "$CFG" \
  training.output_dir=/tmp/p3out training.val_every=2 training.log_every=1 \
  training.val_max_batches=12 debug.enabled=true debug.max_sequences=8 "$@" 2>&1 \
  | grep --line-buffered -vE "Loaded GT joints|Loaded 2D GT|No calibration for|\[VIS\]"
echo "=== DONE rc=$? @ $(date) ==="
