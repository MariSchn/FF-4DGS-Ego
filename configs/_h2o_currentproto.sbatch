#!/bin/bash
#SBATCH --account=3dv
#SBATCH --partition=jobs
#SBATCH --gpus=gb10:1
#SBATCH --time=04:00:00
# ---------------------------------------------------------------------------
# H2O packed-npz -> current-protocol store converter (CPU work; the gb10 node is
# requested only because venv_gb10 is aarch64 and the --validate gates import the
# diffsynth/train_hand_head stack). Usage ($HOME is inode-exhausted -> logs to
# scratch joblogs, see memory):
#   mkdir -p /work/scratch/dmonopoli/joblogs
#   sbatch --job-name=h2o_proto \
#     --output=/work/scratch/dmonopoli/joblogs/h2o_proto_%j.out \
#     --error=/work/scratch/dmonopoli/joblogs/h2o_proto_%j.err \
#     configs/_h2o_currentproto.sbatch [max_seqs] [extra converter args...]
#
# Scratch is quota-tight: default converts 8 sequences per run and the converter
# resumes-by-skipping, so just resubmit until "0 converted". Override src/dst via
# H2O_SRC / H2O_DST env vars. Exit code is non-zero on ANY convert or gate
# failure — do not train on the output of a red run.
# ---------------------------------------------------------------------------
set -uo pipefail
SRC="${H2O_SRC:-/work/scratch/dmonopoli/h2o_packed}"
DST="${H2O_DST:-/work/scratch/dmonopoli/h2o_currentproto}"
MAX="${1:-8}"
shift || true
echo "=== H2O->currentproto JOB ${SLURM_JOB_ID:-local} on $(hostname) @ $(date) ==="
echo "=== SRC=$SRC  DST=$DST  MAX_SEQS=$MAX  EXTRA: $* ==="
cd /home/dmonopoli/FF-4DGS-Ego || exit 2
source venv_gb10/bin/activate
echo "PYTHON: $(which python)  ARCH: $(uname -m)"
mkdir -p "$DST"
df -h "$DST" | tail -1
# Home-space guard: converted output is small but the target fs is quota-tight;
# abort before writing if headroom is under ~1GB.
avail_kb=$(df --output=avail "$DST" | tail -1 | tr -d ' ')
if [ "$avail_kb" -lt 1000000 ]; then echo "!!! <1GB free on $DST - abort"; exit 3; fi
python -u -m scripts.preprocessing.h2o_to_currentproto \
  --h2o "$SRC" --out "$DST" \
  --mano_model models/MANO \
  --max_seqs "$MAX" --validate "$@"
RC=$?
echo "=== output size: $(du -sh "$DST" 2>/dev/null | cut -f1)  seqs: $(ls -d "$DST"/*/ 2>/dev/null | wc -l) ==="
echo "=== H2O->currentproto JOB ${SLURM_JOB_ID:-local} rc=$RC COMPLETE @ $(date) ==="
exit $RC
