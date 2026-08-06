#!/bin/bash
#SBATCH --job-name=zguard
#SBATCH --account=3dv
#SBATCH --partition=jobs
#SBATCH --gpus=gb10:1
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/home/dmonopoli/joblogs/zguard_%j.out
#SBATCH --error=/home/dmonopoli/joblogs/zguard_%j.out
#
# TASK #63: WHY IS THE SOLVED SCENE SCALE 42% TOO SMALL, and which of two mechanisms fixes it?
#
# THE MEASUREMENT THAT DEFINES THE PROBLEM. ours_fix59_seg30.json, 1884 segments:
#     s_gt   median 1.0230     (true camera scale, from GT extrinsics)
#     s_hand median 0.6208     (what we solve)
#     ratio  median 0.578      -> our camera translations are 42% too short
#     296/1884 = 15.7% of segments sit exactly on the 0.1 clamp floor
# The hand depth is NOT the culprit: C-MPJPE absolute is ~36mm at ~0.7m, about 5%. So d_scene at
# the sampled pixels reads roughly 1.7x too far. This matters more than it looks: s multiplies the
# camera TRANSLATION, and the W-MPJPE decomposition says translation is the dominant lever
# (lw60: GT translation -47.8mm vs GT rotation -18.8mm).
#
# TWO CANDIDATE MECHANISMS, both one-directional, tested here as a 2x2.
#   H1 BEHIND-CAMERA JOINTS. project_joints_to_norm_pixels clamps depth to Z_MIN=5cm for the
#      projection but returns the RAW depth, so a joint behind the camera lands at a plausible
#      in-frame pixel and contributes a NEGATIVE ratio that the median eats. --require_positive_z
#   H2 BACKGROUND CONTAMINATION. At a hand pixel the nearest visible surface IS the hand, so any
#      misregistration or silhouette blur blends in BACKGROUND, which is strictly FARTHER. d_scene
#      can be pushed up and essentially never down. Under one-sided contamination a low-order
#      statistic is the correct estimator, not the mean-like bilinear blend.
#      --scene_depth_window 3 --scene_depth_reduce min
#
# H2 is the one that predicts the SIZE of the observed bias: hand at 0.5m against background at
# 0.85m is a ratio of 1.7, which is what we measure. H1 predicts the clamp-floor tail. They are
# not exclusive, hence the 2x2 rather than a shootout.
#
# WIN CONDITION, in priority order. W-MPJPE is primary because it is the metric we actually care
# about AND it is computed on every segment; scale_ratio_med is corroboration only, since GT
# extrinsics are present for a small minority of segments.
#   1. W-MPJPE goes DOWN                       <- primary, all segments
#   2. scale_ratio_med moves toward 1.0        <- corroboration, GT-backed but few segments
#   3. clamp-floor % goes down
#   4. C_abs and C_rr do NOT move              <- they are camera-frame and MUST be untouched by
#                                                 any scale change. If they move, something else
#                                                 was altered and the arm is void.
# A drop in clamp-floor rate ALONE is not success: excluding joints can reduce failures simply by
# shrinking the population.
set -uo pipefail
source /work/scratch/dmonopoli/venv_gb10/bin/activate
export XDG_CACHE_HOME=/tmp/xc TORCH_HOME=/tmp/th HF_HOME=/tmp/hf MPLCONFIGDIR=/tmp/mpl
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
mkdir -p /tmp/xc /tmp/th /tmp/hf /tmp/mpl /home/dmonopoli/results
cd /home/dmonopoli/FF-4DGS-Ego

# GATE: both mechanisms must be present, and the window estimator must demonstrably recover a near
# surface, or the arms silently collapse to the baseline and the run reads as "nothing helps".
python - <<'PY'
import sys, inspect
sys.path.insert(0, ".")
import torch
from scripts import eval_world_space as E
from diffsynth.auxiliary_models.worldmirror.models.utils.hand_depth_sampling import (
    sample_depth_at_joints)
sig = inspect.signature(E.predict_clip).parameters
for need in ("require_positive_z", "scene_depth_window", "scene_depth_reduce"):
    if need not in sig:
        raise SystemExit(f"FATAL: predict_clip has no {need} - stale checkout, sync first")
# H1 must exclude a behind-camera joint.
z = torch.tensor([[-0.3]]); d = torch.tensor([[0.9]]); f = torch.tensor([[True]])
if bool(E.ratio_validity_mask(z, d, f, require_positive_z=True).any()):
    raise SystemExit("FATAL: require_positive_z does not exclude z<=0")
# H2 must recover the near surface when the read is misregistered onto background.
res = 32; dm = torch.full((1, 1, res, res), 0.85); dm[..., 12:20, 12:20] = 0.50
g = torch.tensor([(20 + 0.5) / res, (16 + 0.5) / res]).view(1, 1, 1, 1, 2)
plain = float(sample_depth_at_joints(dm, g)[0].reshape(-1)[0])
win = float(sample_depth_at_joints(dm, g, window=3, reduce="min")[0].reshape(-1)[0])
if not (win < plain and abs(win - 0.50) < 1e-4):
    raise SystemExit(f"FATAL: window-min did not recover the near surface (plain={plain}, win={win})")
print(f"GATE PASSED: H1 excludes z<=0; H2 recovers {win:.3f} from a contaminated {plain:.3f}")
PY
[ $? -ne 0 ] && { echo "!! gate failed, aborting"; exit 3; }

python - <<'PY'
import yaml
c = yaml.safe_load(open("configs/exp_p4_jitterrob.yaml"))
c["model"]["warm_start_hand_head"] = "/home/dmonopoli/ckpt_backup/jitterrob10ep_best.pt"
c["model"]["enable_gs"] = True   # no gs_depth means no scale solve and no experiment at all
yaml.safe_dump(c, open("/tmp/zguard.yaml", "w"))
print("cfg written (enable_gs=True)")
PY

# 30 sequences, not 157: this needs a matched contrast, not coverage. --max_seqs takes a
# deterministic prefix, so all four arms see the SAME sequences.
NSEQ=30
run_arm () {   # $1=tag  $2=extra flags
  echo "########## ARM $1 :: $2 ##########"
  python -u -m scripts.eval_world_space --config /tmp/zguard.yaml \
    --data_root /work/scratch/dmonopoli/hoi4d_test157 --max_seqs $NSEQ --max_segs 0 \
    --segment_len 30 --wa_short 30 --robust_scale $2 \
    --declare_protocol "task #63 2x2: $1" \
    --out /home/dmonopoli/results/zguard_$1.json
  echo "### ARM_$1_DONE exit=$? ###"
}

run_arm base    ""
run_arm zonly   "--require_positive_z"
run_arm wonly   "--scene_depth_window 3 --scene_depth_reduce min"
run_arm both    "--require_positive_z --scene_depth_window 3 --scene_depth_reduce min"

echo
echo "=================== TASK #63 2x2 READOUT ==================="
python - <<'PY'
import json, os, statistics as st

ARMS = [("base", "neither"), ("zonly", "H1 z>0"), ("wonly", "H2 win3-min"), ("both", "H1+H2")]
print(f"{'arm':22s} {'W':>8} {'WA_s':>8} {'s_med':>8} {'ratio/GT':>9} {'floor%':>7} "
      f"{'C_abs':>8} {'C_rr':>8}")
base = None
for tag, label in ARMS:
    f = f"/home/dmonopoli/results/zguard_{tag}.json"
    if not os.path.exists(f):
        print(f"{label:22s}  MISSING {f}")
        continue
    d = json.load(open(f))
    a = d.get("aggregate", {})
    rows = d.get("per_segment", [])
    floor = sum(1 for r in rows if abs(r.get("s_med", 9) - 0.1) < 1e-6)
    rt = [r["scale_ratio_med"] for r in rows if isinstance(r.get("scale_ratio_med"), (int, float))]
    g = lambda k: (f"{a[k]:8.2f}" if isinstance(a.get(k), (int, float)) else f"{'-':>8}")
    print(f"{label:22s} {g('W_MPJPE')} {g('WA_MPJPE_short')} {g('s_med')} "
          f"{(f'{st.median(rt):9.3f}' if rt else f'{chr(45):>9}')} "
          f"{100*floor/max(len(rows),1):6.1f}% {g('C_MPJPE_abs')} {g('C_MPJPE')}")
    if tag == "base":
        base = a
    elif base is not None:
        for k in ("C_MPJPE_abs", "C_MPJPE"):
            if isinstance(a.get(k), (int, float)) and isinstance(base.get(k), (int, float)):
                if abs(a[k] - base[k]) > 0.05:
                    print(f"    !! {k} MOVED {base[k]:.3f} -> {a[k]:.3f}. It is camera-frame and "
                          f"cannot depend on the scene scale. This arm is VOID.")
print()
print("Adopt an arm only if W goes DOWN and scale_ratio_med moves toward 1.0 (from ~0.578).")
print("A lower clamp-floor % alone is NOT success: excluding joints shrinks the population.")
print("If nothing moves W, the scale is not the binding term and the camera head is the next lever.")
PY
echo "### ZGUARD_ALL_DONE ###"
