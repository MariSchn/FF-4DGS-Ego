#!/bin/bash
# ABLATION CONTROL for run_p4_contact_gb10.sh: identical bounded train+W-eval but
# with the root anchor OFF (configs/exp_p4_suponly.yaml). Compare its W to the
# contact run's 246.1 to isolate the anchor's marginal contribution over the
# kp3d_abs supervision both runs share. Same overrides (grad_accum=1, 50 steps).
#
# Run on a gb10 node (sbatch recommended so output persists):
#   sbatch --account=3dv --partition=jobs --gpus=gb10:1 --mem=48G --time=01:30:00 \
#     --output=/work/scratch/dmonopoli/joblogs/suponly_%j.out \
#     --wrap='bash ~/FF-4DGS-Ego/run_p4_suponly_gb10.sh'
set -uo pipefail

source /work/scratch/dmonopoli/venv_gb10/bin/activate
export XDG_CACHE_HOME=/tmp/xc TORCH_HOME=/tmp/th HF_HOME=/tmp/hf MPLCONFIGDIR=/tmp/mpl
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
mkdir -p /tmp/xc /tmp/th /tmp/hf /tmp/mpl
cd /home/dmonopoli/FF-4DGS-Ego
DR=/work/courses/3dv/team25/data/hot3d_aria/preprocessed_pinhole_f609

OUT=/work/scratch/dmonopoli/joblogs
mkdir -p "$OUT"
LOG="$OUT/suponly_run.log"
exec > >(tee "$LOG") 2>&1
echo "### output -> $LOG | node $(hostname) | $(date) ###"

echo "######## TRAIN p4_suponly (anchor OFF; 50 opt-steps, grad_accum=1) @ $(date) on $(hostname) ########"
python -u -m scripts.train_hand_head --config configs/exp_p4_suponly.yaml \
  training.output_dir=/tmp/rt_suponly training.max_steps=50 training.grad_accum_steps=1 \
  training.kp3d_abs_warmup_steps=8 \
  training.log_every=5 training.val_every=25 training.val_max_batches=8 \
  debug.enabled=true debug.max_sequences=8 2>&1 \
  | grep --line-buffered -vE "Loaded GT joints|Loaded 2D GT|No calibration for|\[VIS\]"

echo "######## build eval config (warm-start the suponly-trained head) ########"
python - <<'PY'
import yaml
c = yaml.safe_load(open("configs/exp_p4_suponly.yaml"))
c["model"]["warm_start_hand_head"] = "/tmp/rt_suponly/hand_head_final.pt"
yaml.safe_dump(c, open("/tmp/eval_suponly.yaml", "w"))
print("wrote /tmp/eval_suponly.yaml -> warm_start /tmp/rt_suponly/hand_head_final.pt")
PY

echo "######## W-EVAL: supervision-only head (same args as contact) ########"
python -u -m scripts.eval_world_space --config /tmp/eval_suponly.yaml \
  --data_root "$DR" \
  --max_seqs 4 --max_segs 2 --segment_len 128 --clip_len 16 --stride 8 --wa_short 16 \
  --out "$OUT/world_eval_suponly.json" 2>&1 \
  | grep --line-buffered -vE "fwd\+lift|Loaded GT joints|Loaded 2D GT|No calibration for|\[VIS\]"
echo "######## DONE @ $(date) ########"
echo "Compare: suponly W-MPJPE pooled vs contact 246.1 vs baseline 308.3 (ceiling re16~50-61)"
