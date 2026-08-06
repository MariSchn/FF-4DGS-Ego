#!/bin/bash
# Single-allocation critical-path test: build torch1.13+cu117 venv, pip CUDA-11.8 nvcc,
# assemble CUDA_HOME, compile droid_backends + lietorch, import them. Fail fast.
set -e
VENV=$TMPDIR/hvenv
export PIP_CACHE_DIR=$TMPDIR/pipcache; mkdir -p "$PIP_CACHE_DIR"
HAWOR=/home/dmonopoli/HaWoR

echo "=== venv + torch ==="
/usr/bin/python3.10 -m venv "$VENV"
source "$VENV/bin/activate"
python -m pip install -q --upgrade "pip<24.1" "setuptools<70" wheel
pip install -q torch==1.13.0+cu117 torchvision==0.14.0+cu117 \
    --extra-index-url https://download.pytorch.org/whl/cu117 2>&1 | tail -2
python -c "import torch; print('torch', torch.__version__, torch.version.cuda, 'avail', torch.cuda.is_available())"

echo "=== pip CUDA 11.8 nvcc toolkit ==="
pip install -q nvidia-cuda-nvcc-cu11==11.8.89 nvidia-cuda-runtime-cu11==11.8.89 \
    nvidia-cuda-cccl-cu11==11.8.89.post1 2>&1 | tail -2
NVD=$(python -c "import os,nvidia; print(os.path.dirname(nvidia.__file__))")
echo "NVD=$NVD"; ls "$NVD"
export CUDA_HOME=$TMPDIR/cuda_home
mkdir -p "$CUDA_HOME/bin" "$CUDA_HOME/include" "$CUDA_HOME/lib64"
cp -rn "$NVD/cuda_nvcc/bin/." "$CUDA_HOME/bin/" 2>/dev/null || true
cp -rn "$NVD/cuda_nvcc/nvvm" "$CUDA_HOME/" 2>/dev/null || true
for d in cuda_nvcc cuda_runtime cuda_cccl; do
    [ -d "$NVD/$d/include" ] && cp -rn "$NVD/$d/include/." "$CUDA_HOME/include/" 2>/dev/null || true
    [ -d "$NVD/$d/lib" ]     && cp -rn "$NVD/$d/lib/." "$CUDA_HOME/lib64/" 2>/dev/null || true
done
# torch's cudart headers/libs also help the linker; symlink torch's cuda libs in.
TLIB=$(python -c "import os,torch; print(os.path.join(os.path.dirname(torch.__file__),'lib'))")
ln -sf "$TLIB"/libcudart* "$CUDA_HOME/lib64/" 2>/dev/null || true
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$TLIB:$LD_LIBRARY_PATH"
echo "--- nvcc ---"; nvcc --version | tail -3
echo "--- include count ---"; ls "$CUDA_HOME/include" | wc -l

echo "=== compile droid_backends + lietorch ==="
export TORCH_CUDA_ARCH_LIST="7.5"   # just sm_75 for a fast test build
cd "$HAWOR/thirdparty/DROID-SLAM"
python setup.py install 2>&1 | tail -30
cd "$HAWOR"
echo "=== import test ==="
python -c "import torch, droid_backends, lietorch; print('IMPORT_OK droid_backends + lietorch')"
echo "=== BUILD_TEST_PASS ==="
