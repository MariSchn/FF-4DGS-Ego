#!/bin/bash
#SBATCH --job-name=fpsbd
#SBATCH --account=ls_polle
#SBATCH --output=/cluster/scratch/dmonopoli/fpsbd_%j.out
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=7G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:24g
# Params, FLOPs, latency and peak memory, split inherited versus added, in the three configurations
# that matter. This is the measurement the efficiency review named as the single one that most
# reduces reviewer risk, and it closes four separate defects at once.
#
# It does NOT wait for the five-dataset run. Throughput and arithmetic are properties of the
# architecture, not of the weights, so any checkpoint with the same head configuration gives the
# number the final one will give. The parameter counts are read from the model that was timed.
#
# Three arms, because two of them describe systems we do not run:
#   hands_only            enable_gs off, what fps_probe timed and what produced the 6.59 figure
#   as_scored             enable_gs on + gs_anchor_only, the configuration every world number uses
#   hands_plus_gaussians  full rasterization, what the title promises
#
# Timed at the EVALUATION protocol, clip_len 16 with stride 8. The clips overlap by half, so each
# frame is encoded twice and the deployed throughput is stride/latency, half of clip_len/latency.
# The old probe used stride 16 and reported the larger number against a baseline's wall-clock rate.
set -uo pipefail
S=/cluster/scratch/dmonopoli
cd $S/FF-4DGS-Ego
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=$S/train_env/bin/python
CKPT=$S/ckpts/jitterrob10ep_best.pt
CFG=$S/FF-4DGS-Ego/configs/_fpsbd.yaml

echo "### FPS_BREAKDOWN ${SLURM_JOB_ID:-nojob} $(hostname) $(date) ###"
$PY -c 'import torch; print("torch",torch.__version__,"gpu",torch.cuda.get_device_name(0))' || exit 1
[ -f "$CKPT" ] || { echo "!! no checkpoint $CKPT"; exit 1; }

$PY scripts/_make_eval_cfg.py configs/train_abs2x2_absON_trON.yaml "$CKPT" "$CFG" || exit 2

$PY -u scripts/fps_breakdown.py \
  --config "$CFG" \
  --data_root $S/hoi4d_test157_detv3 \
  --clip_len 16 --stride 8 \
  --warmup 5 --iters 30 \
  --out $S/results/fps_breakdown.json
RC=$?
[ $RC -eq 0 ] || { echo "!! fps_breakdown failed rc=$RC"; exit $RC; }

echo "===== SUMMARY ====="
$PY - <<'PY'
import json
d = json.load(open('/cluster/scratch/dmonopoli/results/fps_breakdown.json'))
print(f'{"arm":22s} {"clip_fps":>9s} {"fps@s8":>8s} {"lat_ms":>8s} {"peak_GiB":>9s} '
      f'{"total_M":>8s} {"train_M":>8s}')
for k in ("hands_only", "as_scored", "hands_plus_gaussians"):
    r = d.get(k)
    if not isinstance(r, dict) or "error" in r:
        print(f'{k:22s}  {r.get("error", "MISSING") if isinstance(r, dict) else "MISSING"}')
        continue
    print(f'{k:22s} {r["clip_fps"]:9.2f} {r["fps"]:8.2f} {1000*r["total_s_mean"]:8.1f} '
          f'{r["peak_mem_GiB"]:9.2f} {r["params_total_M"]:8.1f} {r["params_trainable_M"]:8.1f}')
    fl = r.get("flop_split", {})
    if isinstance(fl, dict) and "_total_inherited" in fl:
        inh = fl["_total_inherited"]; add = fl.get("_total_added", {})
        print(f'{"":22s} FLOPs inherited {inh.get("pct", float("nan")):.1f}%  '
              f'added {add.get("pct", float("nan")):.1f}%')
PY
echo "### FPS_BREAKDOWN_DONE ###"
