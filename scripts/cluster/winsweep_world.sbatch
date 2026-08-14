#!/bin/bash
#SBATCH --job-name=winworld
#SBATCH --account=ls_polle
#SBATCH --output=/cluster/scratch/dmonopoli/winworld_%j.out
#SBATCH --time=20:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=7G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:24g
# World metrics for the window-length ablation (Sec. 4, "Ablations on window length").
#
# The W/WA half of that paragraph currently cites runs whose eval config had enable_cam false, so
# the world lift used identity camera poses and every scale variant collapsed to one number. The
# camera-frame half of the paragraph is unaffected and stands.
#
# These are the TRAIN-MATCHED arms: all three checkpoints are HOI4D-only under one recipe, which
# the paragraph's current 16-frame source (a three-dataset mixture) is not.
set -uo pipefail
S=/cluster/scratch/dmonopoli
cd $S/FF-4DGS-Ego
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=$S/train_env/bin/python

$PY -c 'import torch; print("torch",torch.__version__,"cuda",torch.cuda.is_available())' || exit 1

run_arm () {
  local TAG=$1 SRC=$2 CK=$3
  [ -f "$CK" ] || { echo "!! no checkpoint $CK"; return 1; }
  local EC=$S/FF-4DGS-Ego/configs/_winworld_${TAG}.yaml
  $PY scripts/_make_eval_cfg.py "configs/$SRC" "$CK" "$EC" || return 2
  grep -qE '^\s*enable_cam:\s*true' "$EC" || { echo "!! enable_cam not true for $TAG"; return 3; }
  for SEG in 100 30; do
    echo "===== $TAG segment_len $SEG ====="
    local LOG=$S/winworld_${TAG}_seg${SEG}_$SLURM_JOB_ID.log
    $PY -u -m scripts.eval_world_space --config "$EC" \
      --data_root $S/hoi4d_test157_detv3 \
      --segment_len $SEG --wa_short 30 \
      --out $S/results/winworld_${TAG}_seg${SEG}.json > "$LOG" 2>&1
    local RC=$?
    tail -20 "$LOG"
    [ $RC -eq 0 ] || { echo "!! $TAG seg$SEG failed rc=$RC, see $LOG"; return $RC; }
  done
}

run_arm w16 train_ctrl_hoi4d_only.yaml $S/checkpoints/ctrl_hoi4d_only/latest.pt || exit 4
run_arm w32 train_hoi4d_32f.yaml       $S/checkpoints/hoi4d_32f/latest.pt      || exit 5
run_arm w64 train_hoi4d_64f.yaml       $S/checkpoints/hoi4d_64f/latest.pt      || exit 6

echo '===== WINDOW SWEEP, WORLD METRICS ====='
$PY - <<'PY'
import json, glob, sys, os, re
bad = False
print(f'{"arm":5s} {"seg":>4s} {"W":>8s} {"WA_short":>9s} {"WA_long":>8s} {"C_abs":>7s} {"s":>6s}  degenerate')
for tag in ('w16', 'w32', 'w64'):
    for seg in (100, 30):
        f = f'/cluster/scratch/dmonopoli/results/winworld_{tag}_seg{seg}.json'
        if not os.path.exists(f):
            print(f'{tag:5s} {seg:4d}   MISSING'); bad = True; continue
        d = json.load(open(f)); a = d['aggregate']; ps = d['per_segment']
        ident = sum(1 for r in ps if r.get('W_MPJPE_s1') == r.get('W_MPJPE_spool'))
        deg = ident == len(ps)
        bad = bad or deg
        print(f'{tag:5s} {seg:4d} {a["W_MPJPE_spool"]:8.1f} {a["WA_MPJPE_short_spool"]:9.1f} '
              f'{a["WA_MPJPE_long_spool"]:8.1f} {a["C_MPJPE_abs"]:7.1f} {a["s_pool"]:6.3f}  '
              f'{"YES " + str(ident) + "/" + str(len(ps)) if deg else "no"}')
sys.exit(7 if bad else 0)
PY
[ $? -eq 0 ] || { echo "!! a degenerate or missing arm, not a result"; exit 7; }
echo '### WINWORLD_DONE ###'
