#!/bin/bash
#SBATCH --job-name=oak_conv
#SBATCH --account=ls_polle
#SBATCH --output=/cluster/scratch/dmonopoli/oak_conv_%j.log
#SBATCH --time=120:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=6g
# OakInk2 full conversion, all 627 sequences. For each tar: extract only the ego-serial frames,
# convert that one sequence, then delete the extracted images, because the tars are ~571 GB and
# cannot all be unpacked at once. Resumable: a sequence whose cache already exists is skipped, so
# a timeout is fixed by resubmitting.
#
# Runs on train_env, not conda_hp310. The 2026-08-13 scratch wipe left conda_hp310 without
# typing_extensions and without a working pip, so torch does not import there. train_env carries
# the whole converter dependency set.
set -uo pipefail
S=/cluster/scratch/dmonopoli
PY=$S/train_env/bin/python
IMG=$S/oakink2_img
MANO=$S/arctic/data/body_models
OUT=$S/oakink2_ours

echo "### OAKINK2_CONV ${SLURM_JOB_ID:-nojob} $(hostname) $(date) ###"
[ -x "$PY" ] || { echo "!! interpreter missing: $PY"; exit 1; }
$PY -c 'import torch,numpy,cv2,smplx,trimesh,scipy' || { echo "!! converter deps missing"; exit 1; }
[ -d "$MANO/mano" ] || { echo "!! MANO missing at $MANO"; exit 1; }

mkdir -p $IMG /tmp/oak_anno $OUT
TARS=$(ls $S/oakink2/data/*.tar 2>/dev/null)
NT=$(echo "$TARS" | wc -w)
BEFORE=$(ls $OUT 2>/dev/null | wc -l)
echo "tars: $NT   already converted: $BEFORE"
[ "$NT" -gt 0 ] || { echo "!! no tars, the download did not land"; exit 2; }

i=0; ok=0; skip=0; fail=0
for T in $TARS; do
  i=$((i+1)); seq=$(basename $T .tar); seqid=${seq//\//_}
  if [ -f "$OUT/$seqid/hand_data/gt_joints_cache_cam_v2.pt" ]; then skip=$((skip+1)); continue; fi
  SER=$($PY -c "
import pickle
d = pickle.load(open('$S/oakink2/anno_preview/$seq.pkl','rb'))
cd = d.get('cam_def', {})
ego = [k for k, v in cd.items() if 'ego' in str(v).lower()]
print(ego[0] if ego else '104422070969')" 2>/dev/null)
  rm -rf $IMG/$seq; mkdir -p $IMG/$seq
  tar -xf $T -C $IMG "$seq/$SER" 2>/dev/null || tar -xf $T -C $IMG --wildcards "*/$SER/*" 2>/dev/null || true
  rm -f /tmp/oak_anno/*.pkl; ln -sf $S/oakink2/anno_preview/$seq.pkl /tmp/oak_anno/$seq.pkl
  cd $S/FF-4DGS-Ego
  $PY -m scripts.oakink2_to_ours --anno_root /tmp/oak_anno --image_root $IMG \
    --mano_dir $MANO --out_root $OUT --ego_serial $SER 2>&1 | grep -aE "^\[|SEQ_SKIP|SEQ_FAIL" | tail -2
  rm -rf $IMG/$seq
  if [ -f "$OUT/$seqid/hand_data/gt_joints_cache_cam_v2.pt" ]; then
    ok=$((ok+1)); echo "OK [$i/$NT] $seqid"
  else
    fail=$((fail+1)); echo "FAIL [$i/$NT] $seqid"
  fi
  if [ $((i % 25)) -eq 0 ]; then echo "  progress ok=$ok skip=$skip fail=$fail"; df -h $S | tail -1; fi
done

AFTER=$(ls $OUT 2>/dev/null | wc -l)
echo "OAKINK2_CONV summary ok=$ok skip=$skip fail=$fail of $NT, sequences on disk $BEFORE -> $AFTER"
# A conversion that adds nothing is a failure, not a no-op.
[ "$AFTER" -gt "$BEFORE" ] || { echo "!! no new sequences converted"; exit 3; }
$PY $S/verify_dataset.py $OUT 2>&1 | head -4
echo "### OAKINK2_CONV_DONE $(date) ###"
