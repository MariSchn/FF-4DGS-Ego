#!/bin/bash
#SBATCH --job-name=zguard
#SBATCH --account=3dv
#SBATCH --partition=jobs
#SBATCH --gpus=gb10:1
#SBATCH --mem=48G
#SBATCH --time=8:00:00
#SBATCH --output=/home/dmonopoli/joblogs/zguard_%j.out
#SBATCH --error=/home/dmonopoli/joblogs/zguard_%j.out
#
# TASK #63 A/B: does excluding behind-camera joints from the scene-scale solve fix the segments
# that land on the 0.1 clamp floor?
#
# WHY THIS EXPERIMENT EXISTS. Measured on geo59 (job 104348, post-#59 geometry fix), the first
# 152 scored segments still put 28 of them (18.4%) exactly on the clamp floor, versus 15.7%
# before the fix. So #59 did not close #63. Separately, the surviving s_med values top out at
# 0.982 against a true camera scale of ~1.0, i.e. the median is being pulled DOWN.
#
# THE MECHANISM UNDER TEST. project_joints_to_norm_pixels clamps depth to Z_MIN=0.05 m for the
# projection but returns the RAW depth. A joint 30 cm behind the camera is therefore projected as
# if it sat 5 cm in front, lands at a plausible in-frame pixel, and contributes a NEGATIVE ratio
# z/d_scene that the median consumes. in_frame cannot see it. Proven on CPU in
# tests/test_negative_z_scale_solve.py.
#
# THE TWO ARMS ARE IDENTICAL except for --require_positive_z. Same checkpoint, same sequences,
# same segment length, same seeds. The behind-camera RATE is printed under both arms, so if it
# comes back ~0% the hypothesis is dead and the 18.4% has another cause. Record that either way.
#
# READOUT, in priority order:
#   1. behind-camera % of accepted correspondences  (is the contaminant even there?)
#   2. clamp-failure %                              (does the guard reduce it?)
#   3. s_med                                        (does it move toward 1.0?)
#   4. W / WA                                       (does it not get worse?)
# The guard is adopted only if 2, 3 and 4 all hold. A drop in the failure rate alone is not
# enough: excluding joints could reduce failures by simply shrinking the population.
set -uo pipefail
source /work/scratch/dmonopoli/venv_gb10/bin/activate
export XDG_CACHE_HOME=/tmp/xc TORCH_HOME=/tmp/th HF_HOME=/tmp/hf MPLCONFIGDIR=/tmp/mpl
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
mkdir -p /tmp/xc /tmp/th /tmp/hf /tmp/mpl /home/dmonopoli/results
cd /home/dmonopoli/FF-4DGS-Ego

# GATE: refuse to run on a stale checkout. Without this the job would happily run the OLD code
# and produce two identical arms, which reads as "the guard does nothing" rather than "the guard
# never shipped".
python - <<'PY'
import sys, inspect
sys.path.insert(0, ".")
from scripts import eval_world_space as E
if not hasattr(E, "ratio_validity_mask"):
    raise SystemExit("FATAL: eval_world_space has no ratio_validity_mask - stale checkout, sync first")
sig = inspect.signature(E.predict_clip).parameters
for need in ("steps_out", "require_positive_z"):
    if need not in sig:
        raise SystemExit(f"FATAL: predict_clip has no {need} - stale checkout, sync first")
import torch
# The guard must actually exclude a behind-camera joint, not just accept the kwarg.
z = torch.tensor([[-0.3]]); d = torch.tensor([[0.9]]); f = torch.tensor([[True]])
if not bool(E.ratio_validity_mask(z, d, f, require_positive_z=False).all()):
    raise SystemExit("FATAL: unguarded mask rejects the fixture - test premise broken")
if bool(E.ratio_validity_mask(z, d, f, require_positive_z=True).any()):
    raise SystemExit("FATAL: require_positive_z does NOT exclude a behind-camera joint")
print("GATE PASSED: require_positive_z is present and excludes z<=0")
PY
[ $? -ne 0 ] && { echo "!! gate failed, aborting"; exit 3; }

python - <<'PY'
import yaml
c = yaml.safe_load(open("configs/exp_p4_jitterrob.yaml"))
c["model"]["warm_start_hand_head"] = "/home/dmonopoli/ckpt_backup/jitterrob10ep_best.pt"
c["model"]["enable_gs"] = True   # REQUIRED: no gs_depth means no scale solve and no A/B at all
yaml.safe_dump(c, open("/tmp/zguard.yaml", "w"))
print("cfg written (enable_gs=True)")
PY

# 40 sequences, not 157: the A/B needs a matched contrast, not coverage. Both arms see the same
# 40 because --max_seqs takes a deterministic prefix of the discovered order.
NSEQ=40
for ARM in off on; do
  EXTRA=""
  [ "$ARM" = "on" ] && EXTRA="--require_positive_z"
  echo "########## ARM require_positive_z=$ARM ##########"
  python -u -m scripts.eval_world_space --config /tmp/zguard.yaml \
    --data_root /work/scratch/dmonopoli/hoi4d_test157 --max_seqs $NSEQ --max_segs 0 \
    --segment_len 30 --wa_short 30 --robust_scale $EXTRA \
    --declare_protocol "task #63 A/B, behind-camera joints ${ARM} the scale solve" \
    --out /home/dmonopoli/results/zguard_${ARM}.json
  echo "### ARM_${ARM}_DONE exit=$? ###"
done

echo
echo "=================== TASK #63 A/B READOUT ==================="
python - <<'PY'
import json, os, re

LOG = os.environ.get("SLURM_JOB_ID", "")
rows = {}
for arm in ("off", "on"):
    f = f"/home/dmonopoli/results/zguard_{arm}.json"
    if not os.path.exists(f):
        print(f"{arm:4s} MISSING {f}")
        continue
    a = json.load(open(f)).get("aggregate", {})
    rows[arm] = a

def g(a, k):
    v = a.get(k)
    return f"{v:.4f}" if isinstance(v, (int, float)) else "-"

print(f"{'arm':6s} {'n_seg':>6s} {'s_med':>8s} {'s_pool':>8s} {'W':>8s} {'WA_s':>8s} {'C_abs':>8s} {'C_rr':>8s}")
for arm in ("off", "on"):
    if arm not in rows:
        continue
    a = rows[arm]
    print(f"{arm:6s} {str(a.get('n_segments','-')):>6s} {g(a,'s_med'):>8s} {g(a,'s_pool'):>8s} "
          f"{g(a,'W_MPJPE'):>8s} {g(a,'WA_MPJPE_short'):>8s} "
          f"{g(a,'C_MPJPE_abs'):>8s} {g(a,'C_MPJPE'):>8s}")

if "off" in rows and "on" in rows:
    so, sn = rows["off"].get("s_med"), rows["on"].get("s_med")
    if isinstance(so, (int, float)) and isinstance(sn, (int, float)):
        print(f"\ns_med {so:.4f} -> {sn:.4f}  (true camera scale is ~1.0; "
              f"{'CLOSER' if abs(sn-1.0) < abs(so-1.0) else 'NOT closer'})")
    if so == sn:
        print("!! IDENTICAL s_med: either the guard excluded nothing (behind-camera rate ~0, "
              "hypothesis DEAD) or it never ran. Check the behind-camera % in the [scale] lines.")
PY

echo
echo "--- clamp-floor and behind-camera rates, straight from the per-segment lines ---"
python - <<'PY'
import os, re
log = f"/home/dmonopoli/joblogs/zguard_{os.environ.get('SLURM_JOB_ID','')}.out"
if not os.path.exists(log):
    raise SystemExit(f"log not found: {log}")
txt = open(log, errors="replace").read()
# Split on the arm banners so each arm's segments are counted separately.
parts = re.split(r"########## ARM require_positive_z=(\w+) ##########", txt)
for i in range(1, len(parts) - 1, 2):
    arm, body = parts[i], parts[i + 1]
    s_med = [float(m) for m in re.findall(r"s med/pool=([0-9.]+)/", body)]
    floor = sum(1 for v in s_med if abs(v - 0.1) < 1e-6)
    bc = [float(m) for m in re.findall(r"behind-camera joints ([0-9.]+)% of accepted", body)]
    n = len(s_med)
    print(f"arm={arm:4s} segments={n:5d} on-floor={floor:4d} ({100*floor/max(n,1):5.1f}%) "
          f"s_med_median={sorted(s_med)[n//2] if n else float('nan'):.4f} "
          f"behind-camera_mean={(sum(bc)/len(bc) if bc else 0.0):5.2f}%")
PY
echo "### ZGUARD_ALL_DONE ###"
