#!/bin/bash
#SBATCH --job-name=oak_dl
#SBATCH --account=ls_polle
#SBATCH --output=/cluster/scratch/dmonopoli/oak_dl_%j.log
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4g
# OakInk2 full release from HuggingFace. We hold 109 converted sequences of 627 because an earlier
# download died inside scene_01, and the tars were deleted afterwards by the stream-and-delete
# pattern, so this starts from nothing on disk.
#
# Two failures this replaces, both of which exited 0 having downloaded nothing:
#   - the old script pointed at hawor_env, which the 2026-08-13 scratch wipe removed
#   - set -uo pipefail without -e lets a dead interpreter pass straight through
# Hence the reachability probe and the file-count guard below. Compute nodes reach the internet
# only through eth_proxy, and that proxy has returned 403 for HuggingFace before, so the probe
# runs first and fails the job in seconds rather than after a day of silence.
set -uo pipefail
module load eth_proxy 2>/dev/null || true
S=/cluster/scratch/dmonopoli
PY=$S/train_env/bin/python
export HF_HOME=$S/hf XDG_CACHE_HOME=$S/xdg_cache TMPDIR=$S/tmp HF_HUB_DISABLE_XET=1
mkdir -p $S/oakink2 $S/hf $S/tmp

echo "### OAKINK2_DL ${SLURM_JOB_ID:-nojob} $(hostname) $(date) ###"
[ -x "$PY" ] || { echo "!! interpreter missing: $PY"; exit 1; }

BEFORE=$(ls $S/oakink2/data/*.tar 2>/dev/null | wc -l)
echo "tars on disk before: $BEFORE"

$PY - <<'PY' || exit 2
from huggingface_hub import HfApi
import sys
try:
    n = len(HfApi().list_repo_files("kelvin34501/OakInk-v2", repo_type="dataset"))
except Exception as e:
    sys.exit(f"HuggingFace unreachable from this node: {type(e).__name__}: {e}")
print(f"reachable, {n} files in repo", flush=True)
PY

$PY - <<'PY'
from huggingface_hub import snapshot_download
p = snapshot_download(repo_id="kelvin34501/OakInk-v2", repo_type="dataset",
                      local_dir="/cluster/scratch/dmonopoli/oakink2",
                      max_workers=8)
print("snapshot ->", p, flush=True)
PY
RC=$?

AFTER=$(ls $S/oakink2/data/*.tar 2>/dev/null | wc -l)
ANNO=$(ls $S/oakink2/anno_preview/*.pkl 2>/dev/null | wc -l)
echo "tars after: $AFTER   anno_preview: $ANNO   rc=$RC"
du -sh $S/oakink2 2>/dev/null

# A download that adds nothing is a failure, whatever the interpreter returned.
[ "$AFTER" -gt "$BEFORE" ] || { echo "!! no new tars, treating as FAILED"; exit 3; }
[ "$AFTER" -ge 600 ] || { echo "!! only $AFTER tars, expected ~633, resubmit to resume"; exit 4; }
echo "### OAKINK2_DL_DONE $(date) ###"
