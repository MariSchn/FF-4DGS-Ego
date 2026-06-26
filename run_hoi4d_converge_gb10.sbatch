#!/bin/bash
#SBATCH --account=3dv
#SBATCH --gpus=gb10:1
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --job-name=hoi4dconv
#SBATCH --output=/work/scratch/dmonopoli/joblogs/hoi4dconv_%j.out
#SBATCH --error=/work/scratch/dmonopoli/joblogs/hoi4dconv_%j.err
# ---------------------------------------------------------------------------
# HOI4D scene-metric CONVERGENCE run (issue #2).
#
# The 20.5 -> 13.4 cm dense-depth run was SIGTERM'd at step 80 with headroom. This
# runs it to convergence so Table 4/6 reports a CONVERGED number (removes the
# "best-so-far / SIGTERM'd" caveat). Pure backbone+GS, hand head OFF, dense GT
# sensor-depth supervision over non-masked scene pixels (train_hoi4d_depth.py).
#
# Two things this wrapper handles that the train script does not:
#   * train_hoi4d_depth takes ONLY --config (no CLI dotted overrides) -> we emit a
#     DERIVED config that redirects output_dir to node-local /tmp (the shared /work
#     quota saves 0-byte checkpoints, see memory) and steadies val (val_max_batches).
#   * the run may hit the wall clock again -> a TERM trap copies the best checkpoint
#     out of /tmp to /work + $HOME BEFORE SLURM kills us (--signal gives 120 s grace).
#     best_depth.pt is also written on every val-improvement during the run.
#
# Submit from the login node:  sbatch run_hoi4d_converge_gb10.sbatch
# ---------------------------------------------------------------------------
set -uo pipefail
source /work/scratch/dmonopoli/venv_gb10/bin/activate
export XDG_CACHE_HOME=/tmp/xc TORCH_HOME=/tmp/th HF_HOME=/tmp/hf MPLCONFIGDIR=/tmp/mpl
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
mkdir -p /tmp/xc /tmp/th /tmp/hf /tmp/mpl
cd /home/dmonopoli/FF-4DGS-Ego
echo "### HOI4D converge | node $(hostname) | $(date) | commit $(git rev-parse --short HEAD) ###"

OUT_TMP=/tmp/hoi4d_depth_conv
PERSIST=/work/scratch/dmonopoli/checkpoints/hoi4d_depth_conv
mkdir -p "$OUT_TMP" "$PERSIST"

# --- wall-clock safety: copy the best checkpoint out before SLURM SIGKILLs us -------
save_best () {
  local CK="$OUT_TMP/best_depth.pt"
  if [ -s "$CK" ]; then
    cp -f "$CK" "$PERSIST/best_depth.pt" 2>/dev/null || true
    cp -f "$CK" "$HOME/hoi4dconv_best_depth.pt" 2>/dev/null || true
    echo "[trap] saved best_depth.pt -> $PERSIST and \$HOME ($(date))"
  else
    echo "[trap] no best_depth.pt yet to save ($(date))"
  fi
}
trap 'echo "[trap] caught TERM"; save_best; exit 0' TERM

# --- NeoVerse base ckpt guard (it is a symlink into team25 that has gone missing before) ---
BASE_CKPT=models/NeoVerse/reconstructor.ckpt
if [ ! -s "$BASE_CKPT" ]; then
  echo "!!! base reconstructor.ckpt missing/empty at $BASE_CKPT — restore from HF (Yuppie1204/NeoVerse) first. Aborting."
  exit 4
fi

# --- derived config: redirect output to /tmp, steady the val estimate ----------------
python - <<PY
import yaml
c = yaml.safe_load(open("configs/exp_hoi4d_depth.yaml"))
c["training"]["output_dir"] = "$OUT_TMP"
c["training"]["val_max_batches"] = 20      # steadier best-tracking than the default 12
c["training"]["val_every"] = 25
c["training"]["log_every"] = 10
# epochs stays at the config value (10); the wall clock + TERM trap bound it, and
# best_depth.pt is saved on every improvement so a wall-stop still keeps the best.
yaml.safe_dump(c, open("/tmp/exp_hoi4d_depth_conv.yaml", "w"))
print("[cfg] derived -> /tmp/exp_hoi4d_depth_conv.yaml | output_dir=$OUT_TMP epochs=%d lr=%s unfreeze=%d"
      % (c["training"]["epochs"], c["training"]["lr"], c["training"].get("unfreeze_last_n_blocks", 4)))
PY

echo ""
echo "######## TRAIN (dense depth, partial unfreeze, to convergence) ########"
python -u -m scripts.train_hoi4d_depth --config /tmp/exp_hoi4d_depth_conv.yaml 2>&1 \
  | grep --line-buffered -vE "Loaded GT joints|Loaded 2D GT|\[VIS\]" || true

# normal completion: persist the best + a final eval banner
save_best
echo ""
echo "######## DONE @ $(date) ########"
echo ">>> Converged checkpoint: $PERSIST/best_depth.pt (and ~/hoi4dconv_best_depth.pt)"
echo ">>> Read the final 'best depth residual ... cm' line above: that is the converged"
echo ">>> Table 4/6 number (replaces the SIGTERM'd 13.4 best-so-far). Target: sub-10 cm."
ls -la "$PERSIST" 2>/dev/null || true
