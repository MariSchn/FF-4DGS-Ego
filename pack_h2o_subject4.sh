#!/bin/bash
# Stream-download + pack H2O SUBJECT-4 (the canonical held-out test subject) into
# .npz on the CPU partition. h2o_packed currently holds only subject1_ego_* (the
# fine-tune set), so the B3 held-out eval found 0 subject4 clips. This adds them.
# Runs on interactive-cpu => CONCURRENT with the GPU (does NOT touch the QOS=1 slot).
# Creds inherited from the submit env ($H2O_USER / $H2O_PASS); never hardcode them.
#SBATCH --job-name=h2opack4
#SBATCH --account=3dv
#SBATCH --partition=interactive-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/work/scratch/dmonopoli/joblogs/%j_h2opack4.out
#SBATCH --error=/work/scratch/dmonopoli/joblogs/%j_h2opack4.out
set -uo pipefail
cd /home/dmonopoli/FF-4DGS-Ego

: "${H2O_USER:?set H2O_USER in the submit env (export H2O_USER=... before sbatch)}"
: "${H2O_PASS:?set H2O_PASS in the submit env (export H2O_PASS=... before sbatch)}"

OUT=/work/scratch/dmonopoli/h2o_packed
URL=https://h2odataset.ethz.ch/data/dataset/subject4_ego_v1_1.tar.gz
echo "=== H2O subject4 pack start $(date) | out=$OUT ==="
echo "    (the tar is streamed, never lands on disk; ~47 inodes added)"

# -f makes curl exit non-zero on a 404 so a wrong version suffix fails loud, not silent.
curl -fsSL -u "$H2O_USER:$H2O_PASS" "$URL" \
  | python3 scripts/pack_h2o.py --out "$OUT" --res 224
RC=${PIPESTATUS[0]}
if [ "$RC" -ne 0 ]; then
  echo "!!! download failed (curl rc=$RC) — if 404, the version suffix differs; try"
  echo "!!! subject4_ego_v1.tar.gz or check the dataset page for the exact filename."
  exit "$RC"
fi

echo "=== DONE $(date) ==="
echo ">>> subject4 npz now present:"
ls -1 "$OUT" | grep -c "^subject4" | sed 's/^/    subject4 seqs: /'
echo ">>> total packed seqs: $(ls -1 "$OUT"/*.npz 2>/dev/null | wc -l)"
echo ">>> next: re-run  sbatch run_b3_h2o_subj4.sbatch  (GPU) to get the held-out numbers."
