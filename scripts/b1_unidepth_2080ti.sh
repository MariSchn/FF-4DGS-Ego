#!/bin/bash
# ---------------------------------------------------------------------------
# B1 stage 2 (2080ti / sm_75): build a node-local cu118 venv, install
# UniDepthV2 (no custom CUDA build), and run eval_unidepth_b1.py on the
# pre-staged bundle. Answers E1: can an off-the-shelf metric-depth FM recover
# metric depth AT THE HAND as well as our in-scene hand anchor (4.5 cm)?
#
# Modeled on the proven WiLoR cu118 recipe (torch 2.3.1+cu118 on 2080ti).
# UniDepthV2 has no custom CUDA op, so a clean pip install works on sm_75.
# Everything node-local (/tmp): $HOME is inode-limited, /work is at quota.
#
#   sbatch scripts/b1_unidepth_2080ti.sh
# ---------------------------------------------------------------------------
#SBATCH --job-name=ff4dgs-b1unidepth
#SBATCH --account=3dv
#SBATCH --partition=jobs
#SBATCH --gpus=2080ti:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/dmonopoli/b1_logs/%j_b1uni.out
#SBATCH --error=/home/dmonopoli/b1_logs/%j_b1uni.out
set -uo pipefail
# NOTE: /work is over the per-user disk quota -- Slurm cannot even open a log file
# there, which kills the batch step instantly (RaisedSignal:53 / CANCELLED, 0 s, no
# log). So logs + the tiny JSON go to $HOME (byte-free; 2 small files won't dent the
# inode quota) and ALL heavy work stays node-local in /tmp.

BUNDLE="/work/scratch/dmonopoli/b1_bundle.pt"   # read-only input (already staged)
WEIGHTS="lpiccinelli/unidepth-v2-vitl14"
OUTDIR="/home/dmonopoli/b1_logs"
OUT="$OUTDIR/b1_unidepth.json"
REPO="$HOME/FF-4DGS-Ego"
mkdir -p "$OUTDIR"

echo "=== B1 UniDepth eval $(date) ==="
echo "node=$(hostname) [$(uname -m)]"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "bundle=$BUNDLE weights=$WEIGHTS out=$OUT"

# --- node-local scratch for env + pip + HF cache ---
NODE_SCRATCH="${TMPDIR:-/tmp/$USER.$SLURM_JOB_ID}"
[ -d "$NODE_SCRATCH" ] || { NODE_SCRATCH="/tmp/$USER.$SLURM_JOB_ID"; mkdir -p "$NODE_SCRATCH"; }
echo "node scratch: $NODE_SCRATCH"; df -h "$NODE_SCRATCH" | tail -1
export PIP_CACHE_DIR="$NODE_SCRATCH/pipcache"
export TMPDIR="$NODE_SCRATCH/tmp"
export HF_HOME="$NODE_SCRATCH/hf"
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR" "$HF_HOME"

VENV="$NODE_SCRATCH/venv_unidepth"
UNIDEPTH_SRC="$NODE_SCRATCH/UniDepth"

echo "=== build venv (python3.10 + torch 2.3.1 cu118 for sm_75) ==="
PY=python3.10
$PY --version
$PY -m venv "$VENV"
source "$VENV/bin/activate"
python -m pip install -q --upgrade pip wheel setuptools
python -m pip install -q torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu118
echo "torch: $(python -c 'import torch;print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "nocuda")')"

echo "=== install UniDepthV2 runtime deps (skip xformers/triton/gradio) ==="
# xformers import in UniDepth is try/excepted -> optional. No custom CUDA op to build.
# unidepth/utils/__init__.py unconditionally imports visualization.py (matplotlib) and
# other utils touch wandb/h5py/tabulate/termcolor/trimesh at import time -> include the
# lightweight pure-python ones so the `from unidepth.models import UniDepthV2` chain loads.
python -m pip install -q "numpy<2" einops timm huggingface-hub "pillow>=10.2.0" \
  scipy opencv-python-headless tqdm matplotlib wandb h5py tabulate termcolor trimesh imageio 2>&1 | tail -3

echo "=== fetch UniDepth source (no-deps, runtime import only) ==="
git clone --depth 1 https://github.com/lpiccinelli-eth/UniDepth "$UNIDEPTH_SRC" 2>&1 | tail -3
# install package metadata without pulling its heavy deps (we supplied them above)
python -m pip install -q --no-deps -e "$UNIDEPTH_SRC" 2>&1 | tail -3

echo "=== sanity: import + one-frame infer ==="
python - "$BUNDLE" <<'PY'
import sys, torch
from unidepth.models import UniDepthV2
print("import UniDepthV2 OK")
m = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vitl14").to("cuda").eval()
print("weights loaded OK")
blob = torch.load(sys.argv[1], map_location="cpu")
clip = blob["clips"][0]
img = clip["rgb"][0].float().to("cuda")   # [3,H,W] in 0-255
K = clip["K"].to("cuda")
with torch.no_grad():
    pred = m.infer(img, K)
d = pred["depth"].squeeze()
print(f"depth map: shape={tuple(d.shape)} min={d.min():.3f} max={d.max():.3f} median={d.median():.3f} m")
print("MILESTONE-1 PASS: UniDepthV2 runs on a B1 frame on sm_75")
PY
if [ $? -ne 0 ]; then echo "MILESTONE-1 FAIL: see error above"; exit 1; fi

echo "=== run full B1 eval over the bundle ==="
cd "$REPO"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python scripts/eval_unidepth_b1.py --bundle "$BUNDLE" --weights "$WEIGHTS" --out "$OUT"
echo "=== done $(date) ==="
echo "RESULT JSON:"; cat "$OUT" 2>/dev/null || echo "(no json written)"
