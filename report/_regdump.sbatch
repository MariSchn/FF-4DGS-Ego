#!/bin/bash
#SBATCH --job-name=regdump
#SBATCH --account=3dv
#SBATCH --partition=jobs
#SBATCH --gpus=gb10:1
#SBATCH --mem=48G
#SBATCH --time=1:00:00
#SBATCH --output=/home/dmonopoli/joblogs/regdump_%j.out
#SBATCH --error=/home/dmonopoli/joblogs/regdump_%j.out
#
# Supervisor request (Cyrus, 2026-08-06): "can we have some visual intermediate results for the
# registration steps?" This dumps the arrays behind R1/R2/R3 so the figure is made from a real
# forward pass instead of drawn by hand.
#
# THIS IS THE SECOND ATTEMPT AND THE REASON MATTERS. The first version of
# dump_registration_steps.py built its own model-load + forward + projection. It drifted from
# eval_world_space in two ways nobody noticed until the output was inspected: no bfloat16
# autocast, and no cond_flags. It produced a CONSTANT hand depth of -0.0205 m on all three
# sequences, i.e. every hand behind the camera, and 100% of "valid" joints had z <= 0. That would
# have gone in front of the supervisor as the method figure.
#
# The script now owns NO forward pass. predict_clip gained a steps_out hook and emits the
# intermediates from inside the eval's own solve, so the panel shows what the eval does by
# construction rather than by coincidence.
#
# Three sequences so the figure has a choice; the script itself prefers a clip whose scale solve
# did NOT hit the clamp (a clamped clip is a task #63 failure case, not a method illustration).
set -uo pipefail
source /work/scratch/dmonopoli/venv_gb10/bin/activate
export XDG_CACHE_HOME=/tmp/xc TORCH_HOME=/tmp/th HF_HOME=/tmp/hf MPLCONFIGDIR=/tmp/mpl
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
mkdir -p /tmp/xc /tmp/th /tmp/hf /tmp/mpl /home/dmonopoli/results
cd /home/dmonopoli/FF-4DGS-Ego

# GATE: the whole point of the rewrite is that the dumper reuses the eval. Refuse to run if this
# checkout still has the standalone version.
python - <<'PY'
import sys, inspect
sys.path.insert(0, ".")
src = open("scripts/dump_registration_steps.py").read()
if "build_model" not in src or "predict_clip" not in src:
    raise SystemExit("FATAL: dumper does not import the eval's build_model/predict_clip - stale checkout")
if "compute_joints_from_batch" in src:
    raise SystemExit("FATAL: dumper still re-implements the projection - stale checkout")
from scripts import eval_world_space as E
if "steps_out" not in inspect.signature(E.predict_clip).parameters:
    raise SystemExit("FATAL: predict_clip has no steps_out hook - stale checkout")
print("GATE PASSED: dumper reuses the eval's forward")
PY
[ $? -ne 0 ] && { echo "!! gate failed, aborting"; exit 3; }

python - <<'PY'
import yaml
c = yaml.safe_load(open("configs/exp_p4_jitterrob.yaml"))
c["model"]["warm_start_hand_head"] = "/home/dmonopoli/ckpt_backup/jitterrob10ep_best.pt"
c["model"]["enable_gs"] = True   # no gs_depth means no scale solve and nothing to visualise
yaml.safe_dump(c, open("/tmp/regdump.yaml", "w"))
print("cfg written (enable_gs=True)")
PY

for IDX in 0 1 2 3 4; do
  echo "########## sequence index $IDX ##########"
  python -u -m scripts.dump_registration_steps \
    --config /tmp/regdump.yaml \
    --data_root /work/scratch/dmonopoli/hoi4d_test157 \
    --seq_index $IDX --n_clips 8 \
    --out /home/dmonopoli/results/regsteps_seq${IDX}.pt
  echo "### SEQ${IDX}_DONE exit=$? ###"
done

echo
echo "=================== WHICH DUMPS ARE USABLE ==================="
python - <<'PY'
import glob, torch
for f in sorted(glob.glob("/home/dmonopoli/results/regsteps_seq*.pt")):
    try:
        d = torch.load(f, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"{f}: UNREADABLE {type(e).__name__}: {e}")
        continue
    p = d["panel"]
    v = p["valid"]
    hz = p["hand_z"][v]
    n = int(v.sum())
    frac_bad = float((hz <= 0).float().mean()) if n else float("nan")
    print(f"{f.split('/')[-1]:22s} clip={p.get('clip_index')} s={p['s']:.4f} "
          f"raw={p['s_raw']:.4f} clamped={p['s_failed']} valid={n:5d} "
          f"hand_z_med={float(hz.median()) if n else float('nan'):+.4f}m "
          f"frac_z<=0={frac_bad:.3f}")
    if n and float(hz.median()) < 0:
        print("    !! hand depth median is NEGATIVE - do NOT use this one for the figure")
PY
echo "### REGDUMP_ALL_DONE ###"
