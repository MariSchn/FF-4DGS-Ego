#!/bin/bash
# Build a self-contained python3.10 venv on node-local disk for HaWoR, including the
# DROID-SLAM + lietorch CUDA extensions. PROVEN RECIPE:
#   - torch 2.7.0+cu128 (the node's only complete nvcc is system CUDA 13.1; pip CUDA
#     wheels ship ptxas but NOT the nvcc frontend, so we must use system nvcc).
#   - system /usr/local/cuda-13.1 as CUDA_HOME; bypass torch's nvcc/torch CUDA version
#     check; trim DROID gencode to sm_75/80/86 (CUDA 13 dropped sm_60/61/70).
#   - The DROID/lietorch .cu sources are pre-patched on disk: <tensor>.type() ->
#     .scalar_type() inside AT_DISPATCH (modern PyTorch removed the implicit conversion).
# Sourced (not exec'd) so the activated venv + CUDA env vars persist to the caller.
set -e

VENV=$1
HAWOR=/home/dmonopoli/HaWoR
export PIP_CACHE_DIR=$TMPDIR/pipcache
mkdir -p "$PIP_CACHE_DIR"

echo "=== [build] python3.10 venv ==="
/usr/bin/python3.10 -m venv "$VENV"
source "$VENV/bin/activate"
python -m pip install -q --upgrade "pip<24.1" "setuptools<70" wheel ninja

echo "=== [build] torch 2.7.0+cu128 ==="
pip install -q torch==2.7.0 torchvision==0.22.0 \
    --index-url https://download.pytorch.org/whl/cu128 2>&1 | tail -2
python -c "import torch; print('torch', torch.__version__, torch.version.cuda, 'avail', torch.cuda.is_available())"

echo "=== [build] system CUDA 13.1 toolchain ==="
export CUDA_HOME=/usr/local/cuda-13.1
export PATH="$CUDA_HOME/bin:${PATH:-}"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
nvcc --version | tail -2
# Neutralise torch's hard CUDA-version-skew check (13.1 nvcc vs 12.8 torch).
CPPEXT=$(python -c "import torch.utils.cpp_extension as e; print(e.__file__)")
python - "$CPPEXT" <<'PY'
import sys
f = sys.argv[1]; lines = open(f).read().splitlines(); out = []; n = 0
for ln in lines:
    if "raise RuntimeError(CUDA_MISMATCH_MESSAGE" in ln:
        out.append(ln[:len(ln) - len(ln.lstrip())] + "pass  # patched: tolerate CUDA skew"); n += 1
    else:
        out.append(ln)
open(f, "w").write("\n".join(out) + "\n"); print("patched", n, "raise(s)")
assert n >= 1
PY

echo "=== [build] chumpy (patched for numpy>=1.24) ==="
pip install -q "numpy<2"
pip install -q --no-build-isolation chumpy || true
CHU=$(python -c "import os,sys; p=[x for x in sys.path if 'site-packages' in x][0]; print(os.path.join(p,'chumpy','__init__.py'))")
[ -f "$CHU" ] && sed -i 's/^from numpy import bool.*/from numpy import nan, inf/' "$CHU"
python -c "import chumpy; print('chumpy', chumpy.__version__, 'OK')" || echo "chumpy import still failing"

echo "=== [build] DROID-SLAM + lietorch (sm_75/80/86) ==="
cd "$HAWOR/thirdparty/DROID-SLAM"
[ -f setup.py.bak ] && cp setup.py.bak setup.py || cp setup.py setup.py.bak
python - <<'PY'
import re
s = open("setup.py").read()
for a in ("60", "61", "70"):
    s = re.sub(rf"\s*'-gencode=arch=compute_{a},code=sm_{a}',", "", s)
open("setup.py", "w").write(s); print("trimmed gencode")
PY
export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6"
BLOG=$TMPDIR/droid_build.log
python setup.py install > "$BLOG" 2>&1 || { echo "DROID build FAILED:"; grep -nE "error:|fatal error" "$BLOG" | head -20; tail -20 "$BLOG"; return 1; }
cd "$HAWOR"
python -c "import torch, droid_backends, lietorch; print('droid_backends + lietorch IMPORT OK')"

echo "=== [build] HaWoR python deps ==="
pip install -q \
    einops xtcocotools pandas hydra-core hydra-colorlog pyrootutils rich \
    webdataset ultralytics pulp supervision pycocotools joblib natsort evo \
    pytorch-minimize "mmengine==0.10.4" plyfile easydict loguru dill lapx \
    opencv-python smplx==0.1.28 timm scikit-image roma yacs 2>&1 | tail -3
pip install -q "numpy<2"
pip install -q pytorch-lightning==2.2.4 --no-deps
pip install -q lightning-utilities "torchmetrics==1.4.0"
# torch-scatter: prebuilt wheel for torch2.7+cu128 if available, else source build.
pip install -q torch-scatter -f https://data.pyg.org/whl/torch-2.7.0+cu128.html 2>&1 | tail -2 || \
    pip install -q --no-build-isolation torch-scatter || echo "torch-scatter FAILED (may be optional)"
pip install -q "numpy<2"

echo "=== [build] pytorch3d (source build vs CUDA 13.1; needed for SLAM hand masks) ==="
# CUDA_HOME / PATH / version-check-patch already in effect from above.
export FORCE_CUDA=1
P3DLOG=$TMPDIR/p3d_build.log
pip install -q "fvcore" "iopath" 2>&1 | tail -1
pip install -q --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@stable" \
    > "$P3DLOG" 2>&1 \
    && python -c "import pytorch3d; print('pytorch3d', pytorch3d.__version__, 'OK')" \
    || { echo "pytorch3d build FAILED:"; grep -nE "error:|fatal error" "$P3DLOG" | head -15; tail -15 "$P3DLOG"; }
pip install -q "numpy<2"

echo "=== [build] DONE ==="
