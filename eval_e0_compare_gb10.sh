#!/bin/bash
# ---------------------------------------------------------------------------
# E0 same-split comparison: all FOUR checkpoints on ONE locked 9-seq val split
# (600 clips even-sampled), so abs-3D's marginal effect is apples-to-apples:
#   baseline   = Marian 50mm (no fine-tune)
#   control    = warm-start + fine-tune, abs-3D & anchor ZEROED  (E0)
#   warmstart  = warm-start + fine-tune + abs-3D + anchor 1.0
#   a01        = warm-start + fine-tune + anchor 0.1 (scene-recovery retune)
# abs-3D is causal IF warmstart << control on the SAME split.
#   sbatch eval_e0_compare_gb10.sh
# ---------------------------------------------------------------------------
#SBATCH --job-name=ff4dgs-e0cmp
#SBATCH --account=3dv
#SBATCH --gpus=gb10:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00

set -euo pipefail

CFG="configs/exp_p2_pinhole_largeval.yaml"
VL="/work/scratch/dmonopoli/eval_out/e0_split.json"   # fresh lock; all 4 share it
OUT=/work/scratch/dmonopoli/eval_out
LIMIT=600
mkdir -p "$OUT"; rm -f "$VL"

source venv_gb10/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1

declare -A CKPTS=(
  [baseline]="/work/courses/3dv/team25/checkpoints/default/hand_head_final.pt"
  [control]="checkpoints/p2_control/best_mpjpe.pt"
  [warmstart]="checkpoints/p2_warmstart/best_mpjpe.pt"
  [a01]="checkpoints/p2_warmstart_a01/best_mpjpe.pt"
)
# baseline first => it locks the split; the rest reuse it.
for name in baseline control warmstart a01; do
  echo "########## ${name}: ${CKPTS[$name]} ##########"
  python3 -m scripts.eval_hand_head --config "${CFG}" --ckpt "${CKPTS[$name]}" \
    --val-list "${VL}" --limit-clips "${LIMIT}" --out "${OUT}/e0_${name}.json" --batch-size 4 || true
done

echo "=== E0 SAME-SPLIT SUMMARY (9-seq, ${LIMIT} clips) ==="
for name in baseline control warmstart a01; do
  f="${OUT}/e0_${name}.json"
  [ -f "$f" ] && python3 -c "import json; d=json.load(open('${f}')); a=d['metrics']['all']; print(f\"${name:10s}: {d['num_clips']} clips  MPJPE={a['MPJPE']:.2f}  PA={a['PA_MPJPE']:.2f}\")" || echo "${name}: (no result)"
done
echo "Finished: $(date)"
