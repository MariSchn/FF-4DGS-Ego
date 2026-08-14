#!/bin/bash
#SBATCH --job-name=scaletest
#SBATCH --account=ls_polle
#SBATCH --output=/cluster/scratch/dmonopoli/scaletest_%j.out
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=7G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:24g
# Cyrus: "So when you remove all the scaling modules, the performance will improve? Ok, testify
# that. If it's useless we just don't use."
#
# The ablation is W_MPJPE (solved scene scale) against W_MPJPE_s1 (the module removed, s=1), with
# W_MPJPE_gtscale as the oracle that says whether a BETTER scale would have helped. All three come
# from one pass, so they share a checkpoint, a box source and a segmentation.
#
# The checkpoint is the HOI4D-trained one: the question is about the scale mechanism, not the
# training mixture, so it does not wait for the five-dataset run.
set -uo pipefail
S=/cluster/scratch/dmonopoli
cd $S/FF-4DGS-Ego
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=$S/train_env/bin/python
CKPT=$S/ckpts/jitterrob10ep_best.pt
[ -f "$CKPT" ] || { echo NO_CKPT; exit 1; }

$PY -c 'import torch; print("torch",torch.__version__,"cuda",torch.cuda.is_available())' || exit 1

EC=$S/FF-4DGS-Ego/configs/_scaletest_eval.yaml
# _make_eval_cfg forces enable_gs (else the scale pins to 1.0) AND enable_cam (else the camera head
# never publishes camera_poses, the world lift falls back to identity poses, and every scale arm
# returns the same number because s multiplies a zero translation). Do not pass --keep_gs_off.
$PY scripts/_make_eval_cfg.py configs/train_abs2x2_absON_trON.yaml "$CKPT" "$EC" || exit 2
grep -qE '^\s*enable_cam:\s*true' "$EC" || { echo "!! enable_cam not true in $EC"; exit 3; }

for SEG in 100 30; do
  echo "===== segment_len $SEG ====="
  # Full log to disk. An earlier version piped this through `tail -8`, which is narrower than the
  # eval's own validity warnings, so a run with degenerate camera poses read as a clean result.
  LOG=$S/scaletest_seg${SEG}_$SLURM_JOB_ID.log
  $PY -u -m scripts.eval_world_space --config "$EC" \
    --data_root $S/hoi4d_test157_detv3 \
    --segment_len $SEG --wa_short 30 \
    --out $S/results/scaletest_seg${SEG}.json > "$LOG" 2>&1
  RC=$?
  tail -25 "$LOG"
  [ $RC -eq 0 ] || { echo "!! eval failed rc=$RC, see $LOG"; exit $RC; }
done

echo '===== THE THREE ARMS ====='
$PY - <<'PY'
import json, glob, sys
bad = False
for f in sorted(glob.glob('/cluster/scratch/dmonopoli/results/scaletest_seg*.json')):
    d = json.load(open(f)); a = d['aggregate']; ps = d['per_segment']
    seg = f.split('seg')[-1].split('.')[0]
    ident = sum(1 for r in ps if r.get('W_MPJPE_s1') == r.get('W_MPJPE_spool'))
    print(f'segment {seg}:  (n={len(ps)} segments, solved s={a["s_pool"]:.3f})')
    for lab, k in (('solved s (today)', 'W_MPJPE_spool'), ('s=1, module removed', 'W_MPJPE_s1'),
                   ('s=GT (oracle)', 'W_MPJPE_gtscale')):
        v = a.get(k)
        print(f'   {lab:22s} {v:8.1f}' if isinstance(v, (int, float)) and v == v
              else f'   {lab:22s}   absent')
    print(f'   identical solved-vs-s1: {ident}/{len(ps)}')
    if ident == len(ps):
        print('   !! DEGENERATE: camera translation is ~zero, this run measures nothing')
        bad = True
sys.exit(4 if bad else 0)
PY
[ $? -eq 0 ] || { echo "!! degenerate run, not a result"; exit 4; }
echo '### SCALETEST_DONE ###'
