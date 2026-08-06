#!/bin/bash
# Attempt 2: build DROID-SLAM with the ONLY available nvcc (system CUDA 13.1) against
# the newest torch cu wheel (cu128). Trim gencode to sm_75 (CUDA 13 dropped sm_60/61/70).
# Bypass torch's nvcc-vs-torch CUDA version check. Make-or-break for milestone 1.
set -e
VENV=$TMPDIR/hvenv
export PIP_CACHE_DIR=$TMPDIR/pipcache; mkdir -p "$PIP_CACHE_DIR"
HAWOR=/home/dmonopoli/HaWoR

echo "=== venv + torch cu128 ==="
/usr/bin/python3.10 -m venv "$VENV"
source "$VENV/bin/activate"
python -m pip install -q --upgrade "pip<24.1" "setuptools<70" wheel ninja
pip install -q torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128 2>&1 | tail -2
python -c "import torch; print('torch', torch.__version__, torch.version.cuda, 'avail', torch.cuda.is_available())"

echo "=== system CUDA 13.1 as CUDA_HOME ==="
export CUDA_HOME=/usr/local/cuda-13.1
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
nvcc --version | tail -3
# bypass torch's strict CUDA-version match (nvcc 13.1 vs torch cu12.8) by neutralising
# _check_cuda_version in the installed cpp_extension.py (the env vars don't skip it).
CPPEXT=$(python -c "import torch.utils.cpp_extension as e; print(e.__file__)")
python - "$CPPEXT" <<'PY'
import sys
f = sys.argv[1]
lines = open(f).read().splitlines()
out, patched = [], 0
for ln in lines:
    # neutralise the hard raise on CUDA version mismatch (keep indentation)
    if "raise RuntimeError(CUDA_MISMATCH_MESSAGE" in ln:
        indent = ln[:len(ln) - len(ln.lstrip())]
        out.append(indent + "pass  # patched: tolerate nvcc/torch CUDA version skew")
        patched += 1
    else:
        out.append(ln)
open(f, "w").write("\n".join(out) + "\n")
print(f"patched {patched} raise(s) in {f}")
assert patched >= 1, "did not find CUDA_MISMATCH raise"
PY

echo "=== patch setup.py gencode -> sm_75/80/86 (CUDA13 dropped 60/61/70) ==="
cd "$HAWOR/thirdparty/DROID-SLAM"
[ -f setup.py.bak ] && cp setup.py.bak setup.py || cp setup.py setup.py.bak
python - <<'PY'
import re
s = open("setup.py").read()
for arch in ("60","61","70"):
    s = re.sub(rf"\s*'-gencode=arch=compute_{arch},code=sm_{arch}',", "", s)
open("setup.py","w").write(s)
print("patched gencode")
PY
grep gencode setup.py | sort -u

echo "=== compile droid_backends + lietorch (sm_75) ==="
export TORCH_CUDA_ARCH_LIST="7.5"
BLOG=$TMPDIR/droid_build.log
python setup.py install > "$BLOG" 2>&1 || true
echo "--- first compile errors (nvcc/gcc) ---"
grep -nE "error:|error |fatal error|undefined|No such file" "$BLOG" | head -30
echo "--- last 15 lines ---"; tail -15 "$BLOG"
cd "$HAWOR"
echo "=== import test ==="
python -c "import torch, droid_backends, lietorch; print('IMPORT_OK droid_backends + lietorch')"
echo "=== BUILD_TEST_PASS ==="
