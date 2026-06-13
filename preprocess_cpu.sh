#!/bin/bash
# ---------------------------------------------------------------------------
# Pinhole-rectify HOT3D sequences on an x86 CPU node (P2 prep).
#
# WHY CPU/x86 (not the gb10 training nodes): projectaria_tools ships no aarch64
# binding (ModuleNotFoundError: _core_pybinds), so the undistort step CANNOT run
# on the ARM gb10 nodes. It needs no GPU, so we run it on the dedicated x86
# interactive-cpu partition -- this also keeps the single training GPU free for
# P1b/P2 (zero contention).
#
# Self-contained: builds a small x86 CPU venv in $HOME (idempotent; reused on
# re-run) and runs the lite, torch-CPU preprocessing. Output goes to the course
# data dir (writable, separate from the /work/scratch per-user quota).
#
#   sbatch preprocess_cpu.sh
# ---------------------------------------------------------------------------
#SBATCH --job-name=ff4dgs-preproc-cpu
#SBATCH --account=3dv
# The 3dv account's job_submit plugin force-injects gpu:1 and pins to 'jobs',
# so interactive-cpu/--gpus=0 are unreachable. Pin an x86 GPU type (1080ti) so
# we never land on an ARM gb10/spark node where projectaria_tools cannot import.
#SBATCH --partition=jobs
#SBATCH --ntasks=1
#SBATCH --gpus=1080ti:1
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail

SRC="${1:-/work/courses/3dv/team25/data/hot3d_aria/sequences}"
DST="${2:-/work/courses/3dv/team25/data/hot3d_aria/preprocessed_pinhole_f609}"
SEQS="${3:-P0001_10a27bf7,P0001_15c4300c,P0001_23fa0ee8,P0001_4bf4e21a,P0001_550ea2ac}"
VENV="${HOME}/venv_x86_preproc"
mkdir -p logs

echo "=================================================="
echo "Job        : ${SLURM_JOB_ID:-?} on $(hostname) [$(uname -m)]"
echo "Source     : ${SRC}"
echo "Dest       : ${DST}"
echo "Sequences  : ${SEQS}"
echo "Venv       : ${VENV}"
echo "Started    : $(date)"
echo "=================================================="

if [[ "$(uname -m)" != "x86_64" ]]; then
  echo "ERROR: not on an x86_64 node ($(uname -m)); projectaria_tools needs x86." >&2
  exit 2
fi

# --- Build the lite CPU venv once (torch-CPU + projectaria + image/IO deps) ---
if [[ ! -f "${VENV}/bin/activate" ]]; then
  echo "[venv] creating ${VENV}"
  python3 -m venv "${VENV}"
  source "${VENV}/bin/activate"
  pip install --upgrade pip
  pip install torch --index-url https://download.pytorch.org/whl/cpu
  pip install numpy scipy opencv-python-headless decord smplx "projectaria_tools==2.1.1"
else
  echo "[venv] reusing ${VENV}"
  source "${VENV}/bin/activate"
fi

# --- Import sanity (fail fast with a clear message if a dep is missing) ------
python3 - <<'PY'
import importlib
for m in ("torch","numpy","scipy","cv2","decord","smplx"):
    importlib.import_module(m)
from projectaria_tools.core.sophus import SE3          # the binding that fails on ARM
from projectaria_tools.core import calibration         # noqa
print("[import-check] all preprocessing deps import OK on", __import__("platform").machine())
PY

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
python3 scripts/preprocessing/preprocess_undistort.py \
  --source_root "${SRC}" \
  --dest_root   "${DST}" \
  --focal 609.78 --rescale_factor 1.5 --resolution 224 \
  --sequences "${SEQS}"

echo "Finished   : $(date)"
