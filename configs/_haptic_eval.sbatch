#!/bin/bash
#SBATCH --job-name=eval_haptic
#SBATCH --account=3dv
#SBATCH --gpus=2080ti:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/dmonopoli/joblogs/eval_haptic_%j.out
#SBATCH --error=/home/dmonopoli/joblogs/eval_haptic_%j.err

set -uo pipefail

echo "=== RUNNING HAPTIC BASELINE EVALUATION @ $(date) ==="
echo "node=$(hostname)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

NODE_SCRATCH="${TMPDIR:-/scratch/$USER}"
[ -d "$NODE_SCRATCH" ] || NODE_SCRATCH="/tmp/$USER.$SLURM_JOB_ID"
mkdir -p "$NODE_SCRATCH"

VENV="$NODE_SCRATCH/venv_haptic"
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

python -m pip install -q opencv-python pytorch_lightning "smplx==0.1.28" timm einops pyyaml scikit-image roma 2>&1 | tail -2
python -m pip install -q "numpy<2" 2>&1 | tail -1

export PYTHONPATH="/home/dmonopoli/FF-4DGS-Ego:/home/dmonopoli/haptic:${PYTHONPATH:-}"

python /home/dmonopoli/FF-4DGS-Ego/scripts/eval_haptic_baseline.py \
  --test_root /home/dmonopoli/hoi4d_test \
  --work_dir /home/dmonopoli/haptic_eval_run \
  --out_json /home/dmonopoli/results/haptic_baseline_eval.json

echo "=== HAPTIC EVALUATION JOB DONE @ $(date) ==="
