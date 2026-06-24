#!/bin/bash
# World-space (W-MPJPE) eval for the P3-WABS head, GS enabled so the renderer
# produces rendered_extrinsics + gs_depth. Warm-starts the DEPLOYED hand head
# (persistent on disk) to reproduce the eval crash and now print full tracebacks.
# Run on a gb10 node (needs the GPU):
#     srun --account=3dv --gpus=gb10:1 --mem=32G --time=00:25:00 --pty bash eval_wabs_w_gb10.sh
set -uo pipefail

source /work/scratch/dmonopoli/venv_gb10/bin/activate
export XDG_CACHE_HOME=/tmp/xc TORCH_HOME=/tmp/th HF_HOME=/tmp/hf MPLCONFIGDIR=/tmp/mpl
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
mkdir -p /tmp/xc /tmp/th /tmp/hf /tmp/mpl
cd /home/dmonopoli/FF-4DGS-Ego

echo "=== W-EVAL (wabs, deployed head, GS on) @ $(date) on $(hostname) ==="
python -u -m scripts.eval_world_space \
  --config configs/exp_p3_wabs_eval.yaml \
  --data_root /work/courses/3dv/team25/data/hot3d_aria/preprocessed_pinhole_f609 \
  --max_seqs 4 --max_segs 2 --segment_len 128 --clip_len 16 --stride 8 --wa_short 16 \
  --out /tmp/world_eval_wabs.json
echo "=== done @ $(date) ==="
