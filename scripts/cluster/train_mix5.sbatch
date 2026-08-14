#!/bin/bash
#SBATCH --job-name=mix5
#SBATCH --account=ls_polle
#SBATCH --output=/cluster/scratch/dmonopoli/mix5_%j.out
#SBATCH --time=120:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:24g
# The five-dataset run. ARCTIC + DexYCB + HOT3D + OakInk2 + TACO, with HOI4D and H2O held out and
# evaluated zero-shot. Every headline number in the paper comes from this.
#
# The guards below are not defensive habit. Each one is a defect this project already paid for,
# and each cost more than the job it protects.
# MEMORY. The first attempt asked 4x7G and the kernel OOM-killed it after 53 minutes,
# right after the initial validation, which is the peak: the train and validation
# loaders coexist there. The dataset preloads per-sequence GT into RAM and every
# dataloader worker forks a copy of it, so five stores at num_workers 2 across three
# loaders is up to six copies. Raised to 8x8G and num_workers dropped to 1; with a
# feature cache the loader only reads one .pt per clip, so workers buy little.
set -uo pipefail
S=/cluster/scratch/dmonopoli
cd $S/FF-4DGS-Ego
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=$S/train_env/bin/python
# Both arms run through THIS script so they cannot drift apart. An ablation whose two arms live in
# two scripts ends up measuring whatever else diverged, which is how the unfreezing ablation once
# measured the learning rate. Usage: sbatch train_mix5.sbatch [run_name] [config]
RUN=${1:-mix5}
CFG=${2:-configs/train_mix5_all.yaml}
CKDIR=$S/checkpoints/$RUN
echo "=== ARM: $RUN  CFG: $CFG ==="

echo "=== GUARDS ==="
$PY -c 'import torch,numpy,cv2,smplx,chumpy; print("env OK torch",torch.__version__,"cuda",torch.cuda.is_available())' || exit 1

# 1. kp3d_abs. Omitting the key does not fail, it trains with no absolute supervision at all and
#    produces a model whose C_abs is bad while C_rr looks fine. That cost two full runs.
$PY - <<PYG
import sys, yaml
cfg = yaml.safe_load(open("$CFG"))
w = cfg.get("loss_weights", {})
a = float(w.get("kp3d_abs", 0.0))
if a != 1.0:
    sys.exit(f"kp3d_abs is {a}, must be 1.0")
roots = cfg["data"]["data_roots"]
names = sorted(r["name"] for r in roots)
if len(roots) != 5 or "hoi4d" in names or "h2o" in names:
    sys.exit(f"data_roots must be the five training sets with HOI4D and H2O held out, got {names}")
print("config OK: kp3d_abs 1.0, roots", names)
PYG
[ $? -eq 0 ] || exit 2

# 2. All five caches present AND complete. A missing cache does not fail on its own: the loader
#    falls back to a full backbone forward and the run silently becomes hours per epoch. A PARTIAL
#    cache is worse, because nothing anywhere reports it. The TACO build died at 17568 of 23134 on
#    a full scratch quota while SLURM recorded it COMPLETED 0:0, so ">100 clips" would have passed
#    and the headline run would have trained on 76% of one dataset. Expected counts, not presence.
#    Counts are MEASURED from each completed build, not derived from sequences times stride. A
#    derived figure blocked this run once: TACO yields 19199 clips, not the 23134 that arithmetic
#    over 2311 sequences suggests, because clips per sequence follow sequence length.
declare -A EXPECT=( [arctic]=11000 [dexycb]=5800 [hot3d]=4900 [oakink2]=15000 [taco]=19199 )
for n in arctic dexycb hot3d oakink2 taco; do
  c=$(ls $S/featcache32/$n 2>/dev/null | wc -l)
  e=${EXPECT[$n]}
  echo "  cache $n: $c clips (expect >= $e)"
  [ "$c" -ge "$e" ] || { echo "!! cache $n has $c clips, expected at least $e, refusing to start"; exit 3; }
done

# A clip truncated by a full disk still satisfies os.path.exists, so the cache builder skips it on
# resume and training loads a zero-byte tensor. Cheap to check, impossible to debug later.
BAD=$(find $S/featcache32 -name '*.pt' -size -1M 2>/dev/null | head -5)
[ -z "$BAD" ] || { echo "!! truncated cache clips present:"; echo "$BAD"; exit 3; }

echo "=== TRAIN ==="
$PY -u -m scripts.train_hand_head --config $CFG
RC=$?; [ $RC -eq 0 ] || { echo "!! train failed rc=$RC"; exit $RC; }

# 3. The checkpoint must have taken steps. best_val_loss.pt once sat at global_step=1 because
#    val_every counts OPTIMIZER steps and grad_accum made the run validate twice; the resulting
#    C-abs of 725 was reported as a finding before anyone printed the step count.
CKPT=$CKDIR/hand_head_final.pt
[ -f "$CKPT" ] || CKPT=$(ls -t $CKDIR/*.pt 2>/dev/null | head -1)
[ -n "$CKPT" ] && [ -f "$CKPT" ] || { echo "!! no checkpoint"; exit 4; }
$PY -c "
import torch,sys
d=torch.load('$CKPT',map_location='cpu')
gs=d.get('global_step', d.get('step'))
print('checkpoint', '$CKPT', 'global_step', gs)
sys.exit(0 if (gs is None or gs>100) else 5)
" || { echo "!! checkpoint barely stepped, not evaluating it"; exit 5; }

# 5. Eval roots must exist BEFORE the days of training, not after. The H2O root read
#    $S/h2o/store, which has never existed: the store is $S/h2o_currentproto. The eval is
#    piped into tail, so its exit status was discarded and the job would have printed
#    MIX5_DONE with half its results missing.
HOI4D_ROOT=$S/hoi4d_test157_detv3
H2O_ROOT=$S/h2o_currentproto
for R in "$HOI4D_ROOT" "$H2O_ROOT"; do
  [ -d "$R" ] || { echo "!! eval root $R does not exist, refusing to train for days first"; exit 7; }
done

echo "=== EVAL: HOI4D-157, zero-shot ==="
# _make_eval_cfg forces enable_gs on. Do NOT pass --keep_gs_off: the training configs set it
# false, and copying that into a world eval makes every W and WA non-metric (scale pinned to 1.0)
# while the job still exits 0. That one bit us three times.
EC=$S/FF-4DGS-Ego/configs/_${RUN}_hoi4d_eval.yaml
$PY scripts/_make_eval_cfg.py $CFG "$CKPT" "$EC"
# 4. enable_cam. Without it the model never publishes camera_poses, the world lift falls back to
#    identity poses, and every W and WA measures hand-joint chaining with a camera that never moves.
#    The training configs set it false, which is correct for training and fatal for a world eval.
for E in "$EC"; do
  grep -qE '^\s*enable_cam:\s*true' "$E" || { echo "!! enable_cam not true in $E"; exit 6; }
done
set -o pipefail
$PY -u -m scripts.eval_world_space --config "$EC" \
  --data_root "$HOI4D_ROOT" \
  --segment_len 100 --wa_short 30 \
  --out $S/results/${RUN}_hoi4d157.json 2>&1 | tail -12
[ $? -eq 0 ] || { echo "!! HOI4D eval failed"; exit 8; }

echo "=== EVAL: H2O, zero-shot ==="
EH=$S/FF-4DGS-Ego/configs/_${RUN}_h2o_eval.yaml
$PY scripts/_make_eval_cfg.py $CFG "$CKPT" "$EH"
grep -qE '^\s*enable_cam:\s*true' "$EH" || { echo "!! enable_cam not true in $EH"; exit 6; }
$PY -u -m scripts.eval_world_space --config "$EH" \
  --data_root "$H2O_ROOT" \
  --segment_len 100 --wa_short 30 \
  --out $S/results/${RUN}_h2o.json 2>&1 | tail -12
[ $? -eq 0 ] || { echo "!! H2O eval failed"; exit 8; }

echo "### MIX5_DONE ###"
