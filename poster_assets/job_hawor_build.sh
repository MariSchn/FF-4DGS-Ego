#!/bin/bash
#SBATCH --job-name=hawor-build
#SBATCH --account=3dv
#SBATCH --partition=jobs
#SBATCH --gpus=2080ti:1
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/home/dmonopoli/hawor_jobs/logs/%j_hawor.out
#SBATCH --error=/home/dmonopoli/hawor_jobs/logs/%j_hawor.out
set -uo pipefail

echo "=== HaWoR build+demo $(date) ==="
echo "node=$(hostname)"; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
nvcc --version | tail -2

HAWOR=/home/dmonopoli/HaWoR
NODE_SCRATCH="${TMPDIR:-/scratch/$USER}"
[ -d "$NODE_SCRATCH" ] || NODE_SCRATCH="/tmp/$USER.$SLURM_JOB_ID"
mkdir -p "$NODE_SCRATCH"
echo "node scratch: $NODE_SCRATCH"; df -h "$NODE_SCRATCH" | tail -1
export PIP_CACHE_DIR="$NODE_SCRATCH/pipcache"
export TMPDIR="$NODE_SCRATCH/tmp"
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"

# /work/scratch is at DISK quota (cannot write) and $HOME is at INODE quota (a venv =
# thousands of small files). So build the venv in NODE-LOCAL scratch each job.
VENV="$NODE_SCRATCH/venv_hawor"
REBUILD="${REBUILD:-0}"

# CUDA 12.0 toolkit + gcc-11 host compiler (gcc-13 too new for nvcc 12.0).
export CC=gcc-11 CXX=g++-11
export CUDA_HOME=/usr
export TORCH_CUDA_ARCH_LIST="7.5"   # 2080ti only -> fast builds
export MAX_JOBS=8

if [ "$REBUILD" = "1" ] || [ ! -f "$VENV/.ready" ]; then
  echo "=== building venv at $VENV ==="
  rm -rf "$VENV"
  python3.10 -m venv "$VENV"
  source "$VENV/bin/activate"
  python -m pip install -q --upgrade pip wheel setuptools
  echo "--- torch 2.1.2 + cu121 (matches nvcc 12.0; sm_75 ok) ---"
  python -m pip install -q torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
  python -c "import torch;print('torch', torch.__version__, 'cuda', torch.version.cuda)"

  echo "--- core deps (skip pytorch3d/torch-scatter/aitviewer for now) ---"
  python -m pip install -q "numpy==1.26.4" "setuptools<70" wheel
  # chumpy (smplx dep) -- install no-build-isolation then patch numpy aliases
  python -m pip install -q --no-build-isolation chumpy 2>&1 | tail -2
  SP="$VENV/lib/python3.10/site-packages"
  python - "$SP" <<'PY'
import sys, glob, re, os
sp = sys.argv[1]; patched = 0
for f in glob.glob(os.path.join(sp, "chumpy", "*.py")):
    txt = open(f).read()
    new = re.sub(r"from numpy import bool[^\n]*", "from numpy import nan, inf", txt)
    for bad, good in [("bool","bool"),("object","object"),("float","float"),
                      ("int","int"),("complex","complex"),("unicode","str"),("str","str")]:
        new = re.sub(r"\bnp\."+bad+r"\b(?![\w])", good, new)
    if new != txt: open(f,"w").write(new); patched += 1
print(f"patched {patched} chumpy file(s)")
PY
  python -c "import chumpy; print('chumpy ok', chumpy.__version__)" || { echo "CHUMPY FAIL"; exit 1; }

  python -m pip install -q \
    opencv-python pyrender scikit-image "smplx==0.1.28" yacs timm einops \
    xtcocotools pandas hydra-core hydra-colorlog pyrootutils rich webdataset \
    "ultralytics==8.1.34" pulp supervision pycocotools joblib natsort \
    evo plyfile easydict loguru dill lapx "mmengine==0.10.4" trimesh \
    "pytorch-lightning==2.2.4" --no-deps 2>&1 | tail -3
  python -m pip install -q lightning-utilities "torchmetrics==1.4.0" omegaconf antlr4-python3-runtime 2>&1 | tail -2
  # re-pin numpy (ultralytics/skimage may bump it)
  python -m pip install -q "numpy==1.26.4" 2>&1 | tail -1
  touch "$VENV/.ready"
else
  echo "=== reusing venv at $VENV ==="
  source "$VENV/bin/activate"
fi

python -c "import torch;print('torch', torch.__version__, 'cuda avail', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NOCUDA')"

# ---------------------------------------------------------------------------
# Build DROID-SLAM
# ---------------------------------------------------------------------------
echo "=== build DROID-SLAM ($(date)) ==="
cd "$HAWOR/thirdparty/DROID-SLAM"
if python -c "import droid_backends, lietorch" 2>/dev/null; then
  echo "droid_backends + lietorch already importable, skip build"
else
  python -m pip install -q ninja
  # build-base -> node-local to keep build object files OUT of $HOME (inode quota)
  python setup.py build --build-base "$NODE_SCRATCH/droid_build" install 2>&1 | tail -45
  echo "DROID setup exit: ${PIPESTATUS[0]}"
fi
cd "$HAWOR"
python -c "import droid_backends, lietorch; print('DROID-SLAM OK')" || { echo "DROID-SLAM IMPORT FAILED"; }

# ---------------------------------------------------------------------------
# pytorch3d (needed by lib/vis/renderer imported in hawor_video)
# ---------------------------------------------------------------------------
echo "=== install pytorch3d ($(date)) ==="
if python -c "import pytorch3d" 2>/dev/null; then
  echo "pytorch3d already present"
else
  python -m pip install -q "git+https://github.com/facebookresearch/pytorch3d.git@stable" 2>&1 | tail -30
fi
python -c "import pytorch3d; print('pytorch3d', pytorch3d.__version__)" || echo "PYTORCH3D IMPORT FAILED"

echo "=== ENV BUILD COMPLETE $(date) ==="
python -m pip list 2>/dev/null | grep -iE "torch|pytorch3d|droid|lietorch|mmcv|mmengine|chumpy|numpy|ultralytics|smplx" || true

# ---------------------------------------------------------------------------
# Milestone 1: run demo on bundled example video (cam mode to skip world viz heavy path)
# ---------------------------------------------------------------------------
echo "=== MILESTONE 1: demo on example/video_0.mp4 ($(date)) ==="
cd "$HAWOR"
export PYOPENGL_PLATFORM=egl
# Run with video in NODE-LOCAL dir so all derived outputs (frames/tracks/SLAM) stay
# off $HOME (inode quota). seq_folder = dirname(video_path).
WORKDIR="$NODE_SCRATCH/hawor_work"
mkdir -p "$WORKDIR"
cp "$HAWOR/example/video_0.mp4" "$WORKDIR/"
python demo.py --video_path "$WORKDIR/video_0.mp4" --vis_mode cam 2>&1 | tail -80
echo "demo exit: ${PIPESTATUS[0]}"
echo "=== outputs produced ==="
find "$WORKDIR" -maxdepth 3 -type d 2>/dev/null | head -20
ls -la "$WORKDIR/video_0/" 2>/dev/null

echo "=== ALL DONE $(date) ==="
