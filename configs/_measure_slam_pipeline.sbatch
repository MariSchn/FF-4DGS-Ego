#!/bin/bash
#SBATCH --job-name=meas_slam_fps
#SBATCH --account=3dv
#SBATCH --gpus=2080ti:1
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=/home/dmonopoli/joblogs/measure_slam_fps_%j.out
#SBATCH --error=/home/dmonopoli/joblogs/measure_slam_fps_%j.err

set -uo pipefail

echo "=== SURGICAL MEASUREMENT OF SLAM PIPELINE FPS @ $(date) ==="
echo "node=$(hostname)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

NODE_SCRATCH="${TMPDIR:-/scratch/$USER}"
[ -d "$NODE_SCRATCH" ] || NODE_SCRATCH="/tmp/$USER.$SLURM_JOB_ID"
mkdir -p "$NODE_SCRATCH"
export PIP_CACHE_DIR="$NODE_SCRATCH/pipcache"
export TMPDIR="$NODE_SCRATCH/tmp"
export HF_HOME="$NODE_SCRATCH/hf"
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR" "$HF_HOME"

VENV="$NODE_SCRATCH/venv_fps_slam_meas"
python3.10 -m venv "$VENV"
source "$VENV/bin/activate"

python -m pip install -q --upgrade pip wheel setuptools
python -m pip install -q torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -q "numpy<2" "setuptools<70" wheel
python -m pip install -q --no-build-isolation chumpy 2>&1 | tail -2

SP="$VENV/lib/python3.10/site-packages"
python - "$SP" <<'PY'
import sys, glob, re, os
sp = sys.argv[1]
for f in glob.glob(os.path.join(sp, "chumpy", "*.py")):
    txt = open(f).read()
    new = re.sub(r"from numpy import bool[^\n]*", "from numpy import nan, inf", txt)
    for bad, good in [("bool", "bool"), ("object", "object"), ("float", "float"),
                      ("int", "int"), ("complex", "complex"), ("unicode", "str"), ("str", "str")]:
        new = re.sub(r"\bnp\." + bad + r"\b(?![\w])", good, new)
    if new != txt:
        open(f, "w").write(new)
PY

python -m pip install -q "smplx==0.1.28" timm einops "ultralytics==8.1.34" opencv-python huggingface_hub scikit-image roma 2>&1 | tail -2
python -m pip install -q "numpy<2" 2>&1 | tail -1
python -m pip install -q --no-deps --no-build-isolation git+https://github.com/warmshao/WiLoR-mini 2>&1 | tail -5

export PYTHONPATH="/home/dmonopoli/FF-4DGS-Ego/models/hamer:${PYTHONPATH:-}"

echo "=== RUNNING SURGICAL MEASUREMENT OF END-TO-END PIPELINE FPS ==="
python /home/dmonopoli/FF-4DGS-Ego/scripts/measure_full_slam_pipeline_fps.py

echo "=== MEASUREMENT DONE @ $(date) ==="
