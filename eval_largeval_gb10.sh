#!/bin/bash
# ---------------------------------------------------------------------------
# Robust hand-MPJPE comparison on a LARGER common val split (val_split 0.1,
# ~9 seqs) instead of the noisy 1-seq default. Evals BOTH checkpoints on the
# SAME locked split (eval_large.json) for an apples-to-apples number:
#   BASELINE = Marian's 50mm hand_head_final.pt
#   OURS     = P2 warm-start best_mpjpe.pt
# The first eval locks the split; the second reuses it.
#   sbatch eval_largeval_gb10.sh
# ---------------------------------------------------------------------------
#SBATCH --job-name=ff4dgs-largeval
#SBATCH --account=3dv
#SBATCH --gpus=gb10:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail

CFG="configs/exp_p2_pinhole_largeval.yaml"
VL="outputs/eval_large.json"
BASELINE="/work/courses/3dv/team25/checkpoints/default/hand_head_final.pt"
OURS="checkpoints/p2_warmstart/best_mpjpe.pt"
mkdir -p logs outputs
rm -f "${VL}"   # fresh lock for the larger split

source venv_gb10/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

# ~9 val seqs => thousands of clips; cap to 600 EVENLY sampled across all seqs
# (eval_hand_head subsamples by stride) for a robust but time-bounded compare.
LIMIT=600

echo "########## BASELINE (no anchor): Marian 50mm — larger val ##########"
python3 -m scripts.eval_hand_head --config "${CFG}" --ckpt "${BASELINE}" \
  --val-list "${VL}" --limit-clips "${LIMIT}" --out outputs/eval_large_baseline.json --batch-size 4

echo "########## OURS: P2 warm-start — larger val (same split) ##########"
python3 -m scripts.eval_hand_head --config "${CFG}" --ckpt "${OURS}" \
  --val-list "${VL}" --limit-clips "${LIMIT}" --out outputs/eval_large_warmstart.json --batch-size 4

echo "=== SUMMARY ==="
for f in outputs/eval_large_baseline.json outputs/eval_large_warmstart.json; do
  python3 -c "import json; d=json.load(open('${f}')); a=d['metrics']['all']; print('${f}:', f\"{d['num_clips']} clips MPJPE={a['MPJPE']:.2f} PA={a['PA_MPJPE']:.2f}\")"
done
echo "Finished: $(date)"
