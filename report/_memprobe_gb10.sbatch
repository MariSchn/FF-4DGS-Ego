#!/bin/bash
#SBATCH --job-name=memprobe
#SBATCH --account=3dv
#SBATCH --partition=jobs
#SBATCH --gpus=gb10:1
#SBATCH --mem=48G
#SBATCH --time=1:00:00
#SBATCH --output=/home/dmonopoli/joblogs/memprobe_%j.out
#SBATCH --error=/home/dmonopoli/joblogs/memprobe_%j.out
#
# TASK #48 FEASIBILITY GATE, moved to gb10.
#
# The variable-length proposal (train n ~ U[2,32], infer 100 frames in ONE pass, drop
# chunk-and-link) rests on an assumption nobody has measured: that the backbone can take 100
# frames at all. Its alternating attention is O((S*P)^2) in frames x patches, so this is not
# obvious, and the whole idea dies if the answer is no.
#
# WHY IT MATTERS MORE NOW. Our seg128 W-MPJPE is 200.9 while the GT-trajectory oracle reaches
# 61.5 (lw60: 40.8), so roughly 140mm of it is camera trajectory, and W is chaining-dominated.
# A single long pass removes intra-segment chaining outright rather than attacking it.
#
# WHY gb10 AND NOT EULER. The Euler copy (job 9892025) has been PENDING for hours behind
# QOSMaxGRESPerUser, and Euler caps gpumem at 24G, which is where this was previously recorded as
# infeasible. gb10 nodes have 120GB unified memory, so the earlier "infeasible" verdict does not
# transfer and has to be re-measured on the hardware we would actually use.
#
# This needs NO trained checkpoint: peak memory depends on architecture and shapes, not weights.
# OOM is caught as a RESULT, not a crash, so the probe reports a ceiling instead of dying.
#
# READ THE OUTPUT AS A GATE:
#   100 frames fit          -> the single-pass variant is worth retraining for.
#   100 fails, 64 fits      -> chunk-and-link stays, but with far fewer, longer chunks.
#   only <=32 fits          -> task #48 is dead in its single-pass form; say so and close it.
set -uo pipefail
source /work/scratch/dmonopoli/venv_gb10/bin/activate
export XDG_CACHE_HOME=/tmp/xc TORCH_HOME=/tmp/th HF_HOME=/tmp/hf MPLCONFIGDIR=/tmp/mpl
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
mkdir -p /tmp/xc /tmp/th /tmp/hf /tmp/mpl /home/dmonopoli/results
cd /home/dmonopoli/FF-4DGS-Ego

echo "### MEMPROBE ${SLURM_JOB_ID:-nojob} | $(hostname) | $(date) ###"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "(no nvidia-smi)"

# Hands-only path first: that is the configuration the single-pass inference would actually use
# (enable_gs off means no Gaussian branch and no rasterization).
python -u -m scripts.probe_frame_memory \
  --config configs/exp_p4_jitterrob.yaml \
  --frames 16,32,48,64,80,100,128 \
  --out /home/dmonopoli/results/memprobe_gb10_handsonly.json
echo "### HANDS_ONLY_DONE exit=$? ###"

# Then WITH the Gaussian branch, because the scene-scale solve needs gs_depth and the world
# metrics are not metric without it. If this one caps out lower than the hands-only run, the
# single-pass variant can still work for camera-frame metrics but not for world ones, and that
# distinction decides what the experiment can claim.
python -u -m scripts.probe_frame_memory \
  --config configs/exp_p4_jitterrob.yaml --enable_gs \
  --frames 16,32,48,64,80,100,128 \
  --out /home/dmonopoli/results/memprobe_gb10_withgs.json
echo "### WITH_GS_DONE exit=$? ###"

echo
echo "=================== TASK #48 GATE ==================="
python - <<'PY'
import json, os
for tag, f in (("hands-only", "/home/dmonopoli/results/memprobe_gb10_handsonly.json"),
               ("with GS   ", "/home/dmonopoli/results/memprobe_gb10_withgs.json")):
    if not os.path.exists(f):
        print(f"{tag}: MISSING {f}")
        continue
    d = json.load(open(f))
    rows = " ".join(f"{r['frames']}:{'OOM' if not r['ok'] else str(r['peak_GiB'])+'G'}"
                    for r in d["rows"])
    print(f"{tag}: max_frames_ok={d['max_frames_ok']:4d} | {rows}")
print()
print("100+ fits        -> single-pass variant is live; retrain with data.random_frames [2,32].")
print("64 fits, 100 not -> keep chunk-and-link but with far fewer, longer chunks.")
print("<=32 only        -> task #48 is dead in single-pass form. Close it and say why.")
PY
echo "### MEMPROBE_ALL_DONE @ $(date) ###"
