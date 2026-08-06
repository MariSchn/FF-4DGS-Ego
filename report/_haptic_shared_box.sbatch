#!/bin/bash
#SBATCH --job-name=hapshared
#SBATCH --account=es_tang
#SBATCH --output=/cluster/scratch/dmonopoli/results/hapshared_%j.log
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4g
#
# TASK #67: give HaPTIC+SLAM an HONEST shared-box world row, at BOTH window lengths.
#
# WHY THIS IS CHEAP. HaPTIC inference on detbox v3 already ran (haptic_detbox.sbatch) and
# haptic_detbox_preds still holds all 157 sequences. Its CAMERA-frame joints are honest; only its
# world_joints are an oracle, because hoi4d_to_haptic.py injected HOI4D GT extrinsics as cTw
# (scripts/haptic_to_worldeval.py:6). So the missing piece is not inference, it is the SE(3)
# composition against a REAL trajectory. That is CPU-only and takes minutes.
#
# WHAT WAS WRONG BEFORE. The paper's HaPTIC+SLAM cell (35.7) came from haptic_slam_preds, which is
# HaPTIC's NATIVE regime, while every other row in that column is shared-box. Separately,
# tab100.sbatch:41 wired haptic_detbox_preds itself into a row labelled "haptic_slam|detbox_v3",
# which is worse: that dir carries the GT-extrinsics oracle, so it put an oracle number under a
# SLAM label. Measured gap between the two on identical hands: W 27.05/WA 23.30 (oracle) vs
# W 42.53/WA 35.71 (composed). The oracle flatters by ~12mm WA.
#
# THE TRAJECTORY IS THE SHARED ONE. --slam_pred_dir $S/hawor_preds is the same DROID-SLAM track
# used by wilor_slam_detbox_preds, mt_slam_detbox_preds and fj2_slam_detbox_preds, so the
# trajectory is not a confound between rows.
#
# JOINT ORDER IS SAFE HERE. haptic_to_worldeval.py remaps via OP2SMPLX16, which
# tests/test_hunt_a.py asserts is identical to preprocess_hoi4d._KPS2D_FOR_SMPLX16. So HaPTIC does
# NOT carry the permutation that contaminated the WiLoR cache (task #65). Checked, not assumed.
set -uo pipefail
S=/cluster/scratch/dmonopoli
ENV=$S/hawor_env
export PATH=$ENV/bin:$PATH
export LD_LIBRARY_PATH=$ENV/lib/python3.10/site-packages/torch/lib:$ENV/lib:${LD_LIBRARY_PATH:-}
export TMPDIR=$S/tmp; mkdir -p $TMPDIR $S/results
cd $S/FF-4DGS-Ego
touch $S/.project-root 2>/dev/null || true
export GIT_CEILING_DIRECTORIES=$S GIT_DISCOVERY_ACROSS_FILESYSTEM=false
echo "### HAPTIC_SHARED_BOX ${SLURM_JOB_ID:-nojob} | $(hostname) | $(date) ###"

# GATE: refuse to run on a checkout without the provenance module, otherwise the output would be
# another unlabelled directory and this whole exercise repeats.
$ENV/bin/python - <<'PY'
import sys, inspect
sys.path.insert(0, ".")
from scripts.pred_provenance import write_provenance, TRAJ_GT_ORACLE, TRAJ_SLAM
from scripts import build_slam_baseline
if "pred_provenance" not in inspect.getsource(build_slam_baseline):
    raise SystemExit("FATAL: build_slam_baseline does not stamp provenance - stale checkout")
print("GATE PASSED: provenance module present and wired into build_slam_baseline")
PY
[ $? -ne 0 ] && { echo "!! gate failed, aborting"; exit 3; }

# 0) Retro-stamp the SOURCE dir, so the composed output inherits a real box_source instead of
#    "INHERITED_UNRECORDED_FROM:...". We know it from haptic_detbox.sbatch, which ran
#    hoi4d_to_haptic.py --box_dir $S/hoi4d_detboxes_v3.
$ENV/bin/python - <<'PY'
import sys; sys.path.insert(0, ".")
from scripts.pred_provenance import write_provenance, TRAJ_GT_ORACLE
S = "/cluster/scratch/dmonopoli"
write_provenance(
    f"{S}/haptic_detbox_preds",
    box_source=f"{S}/hoi4d_detboxes_v3",
    trajectory_source=TRAJ_GT_ORACLE,
    produced_by="scripts/haptic_to_worldeval.py (retro-stamped 2026-08-06 from haptic_detbox.sbatch)",
    n_seqs=157,
    warning=("world_joints use HOI4D GT extrinsics. Camera-frame metrics are honest; W/WA from "
             "this dir are ORACLE. Use haptic_slam_detbox_preds for the world row."),
)
PY

# 1) Compose with the SHARED DROID-SLAM trajectory -> the honest shared-box world row.
echo "===== COMPOSE haptic on detbox v3 + shared HaWoR SLAM traj @ $(date) ====="
$ENV/bin/python -m scripts.build_slam_baseline \
  --cam_pred_dir $S/haptic_detbox_preds \
  --slam_pred_dir $S/hawor_preds \
  --out_dir $S/haptic_slam_detbox_preds 2>&1 | tail -6

# 2) Score at BOTH window lengths. seg30 feeds the short-window table, seg100 the long-window one;
#    wa_short stays 30 in both so the WA column means the same thing across tables.
for SEG in 30 100; do
  echo "===== SCORE haptic_slam_detbox seg${SEG} ====="
  $ENV/bin/python -m scripts.eval_worldspace_baseline --data_root $S/hoi4d_test157_hd \
    --pred_dir $S/haptic_slam_detbox_preds --segment_len $SEG --wa_short 30 \
    --out $S/results/haptic_slam_detbox_seg${SEG}.json 2>&1 | tail -5
done

# 3) The contrast that justifies the whole job: oracle vs composed vs native, same hands.
echo
echo "=================== HaPTIC: ORACLE vs COMPOSED vs NATIVE ==================="
$ENV/bin/python - <<'PY'
import json, os
S = "/cluster/scratch/dmonopoli"
rows = [
    ("oracle GT-traj  (detbox v3)", f"{S}/results/audit_haptic_detbox_preds.json"),
    ("NATIVE + SLAM   (own boxes)", f"{S}/results/audit_haptic_slam_preds.json"),
    ("SHARED + SLAM   (detbox v3)", f"{S}/results/haptic_slam_detbox_seg30.json"),
]
print(f"{'variant':30s} {'W':>8} {'WA_s':>8} {'C_rr':>8} {'C_abs':>9}")
for name, f in rows:
    if not os.path.exists(f):
        print(f"{name:30s}  MISSING {f}"); continue
    a = json.load(open(f)).get("aggregate", {})
    g = lambda k: (f"{a[k]:8.2f}" if isinstance(a.get(k), (int, float)) else f"{'-':>8}")
    print(f"{name:30s} {g('W_MPJPE')} {g('WA_MPJPE_short')} {g('C_MPJPE')} {g('C_MPJPE_abs')}")
print()
print("Expect the SHARED row to be WORSE than the oracle (no GT camera) and to differ from the")
print("native row (different boxes). If SHARED == ORACLE the compose silently kept GT poses;")
print("if SHARED == NATIVE the wrong cam_pred_dir was used. Either means do not use the number.")
PY
echo "### HAPTIC_SHARED_BOX_DONE @ $(date) ###"
