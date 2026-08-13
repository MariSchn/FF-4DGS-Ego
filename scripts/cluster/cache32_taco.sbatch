#!/bin/bash
#SBATCH --job-name=cachetaco
#SBATCH --time=120:00:00
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=7G
#SBATCH --cpus-per-task=4
#SBATCH --output=/cluster/scratch/dmonopoli/cache32_taco_%j.out
# The last of the five feature caches. ARCTIC, DexYCB, HOT3D and OakInk2 are built (~37.6k clips,
# 1.2 TB under featcache32); TACO is what stands between us and a trainable five-dataset pool.
#
# SIZE, measured rather than guessed. 60 sampled sequences average 160 frames, so 2,311 sequences
# are ~370k frames, ~23.1k clips at clip_stride 16, and at the 32 MB per 32-frame clip the existing
# caches actually cost, ~723 GB. Scratch is at 1.6 TB of a 2.70 TB hard quota, so it fits with
# roughly 380 GB to spare. That margin is thin enough that this job checks it as it goes: a cache
# that fills the quota does not merely fail, it takes down every other job writing to scratch.
#
# clip_stride 16 keeps the 50% overlap the other four caches use, so the mixing weights stay
# comparable across roots. Halving it would double the size for no extra coverage.
set -uo pipefail
S=/cluster/scratch/dmonopoli
cd $S/FF-4DGS-Ego-varlen
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=$S/train_env/bin/python

N=taco
R=$S/taco_ours
OUT=$S/featcache32/$N
MIN_FREE_GB=120          # abort below this, leaving room for the 222-clip build and the logs

# The env was rebuilt after the scratch purge killed it, so prove it works before burning a GPU
# allocation on an import error.
$PY -c "import torch, numpy, cv2, smplx, chumpy; print('env OK torch', torch.__version__, 'cuda', torch.cuda.is_available())" \
  || { echo "TRAIN_ENV_BROKEN"; exit 1; }
[ -d "$R" ] || { echo "NO STORE at $R"; exit 1; }
echo "store: $(ls $R | wc -l) sequences"

CFG=$S/FF-4DGS-Ego-varlen/configs/_cache32_${N}.yaml
$PY - "$CFG" <<'PYEOF'
import sys, yaml
cfg = yaml.safe_load(open("configs/train_mix3_kp2d0.yaml"))
cfg["data"]["num_frames"] = 32           # the window is a hard ceiling on variable-length training
cfg["data"].pop("data_roots", None)      # the builder takes a single --data_root
cfg["data"].pop("random_frames", None)   # caches are built at full length, never subsampled
yaml.safe_dump(cfg, open(sys.argv[1], "w"), sort_keys=False)
print("cache cfg num_frames =", cfg["data"]["num_frames"])
PYEOF
[ $? -eq 0 ] || { echo "config gen failed"; exit 2; }

# SMOKE FIRST, on three sequences into a throwaway directory. The previous attempt reached the
# GPU, built the model, and died on the first sample with KeyError: 'hand_poses', because
# taco_to_ours wrote a jsonl schema no other converter uses and verify_box_store never opens the
# jsonl. Three sequences cost two minutes and would have caught it.
SMOKE=$S/featcache32/_taco_smoke
rm -rf "$SMOKE"; mkdir -p "$SMOKE"
$PY -u -m scripts.build_feature_cache \
  --config "$CFG" --data_root "$R" --out "$SMOKE" \
  --clip_stride 16 --batch_size 2 --num_workers 2 --max_seqs 3 2>&1 | tail -12
n_smoke=$(ls "$SMOKE" 2>/dev/null | wc -l)
echo "smoke wrote $n_smoke files"
[ "$n_smoke" -gt 0 ] || { echo "SMOKE PRODUCED NOTHING, not starting the full build"; exit 3; }
rm -rf "$SMOKE"

mkdir -p "$OUT"
echo "########## 32f cache: $N <- $R ##########"
lquota 2>/dev/null | grep scratch | head -1

# Watchdog: kill the build before it fills the quota rather than after. Runs alongside, not after,
# because by the time a build reports failure the disk is already full for everyone else.
( while true; do
    sleep 600
    used_tb=$(lquota 2>/dev/null | awk '/scratch/ && /space/ {print $5}' | head -1)
    free_gb=$(python3 -c "print(int((2.70-float('${used_tb:-0}' or 0))*1024))" 2>/dev/null || echo 999)
    if [ "$free_gb" -lt "$MIN_FREE_GB" ] 2>/dev/null; then
      echo "!!! WATCHDOG: only ${free_gb}GB free, killing the cache build"
      pkill -u $USER -f build_feature_cache
      exit 0
    fi
  done ) &
WATCH=$!

$PY -u -m scripts.build_feature_cache \
  --config "$CFG" --data_root "$R" --out "$OUT" \
  --clip_stride 16 --batch_size 2 --num_workers 3
RC=$?
kill $WATCH 2>/dev/null

echo "files=$(ls "$OUT" 2>/dev/null | wc -l)  size=$(du -sh "$OUT" 2>/dev/null | cut -f1)"
lquota 2>/dev/null | grep scratch | head -1
echo "### CACHE32_TACO_DONE rc=$RC ###"
