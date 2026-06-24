#!/bin/bash
# One-shot Phase-1 CONTACT probe: train the scene-depth root anchor (bounded),
# then world-eval the trained head with the anchor active, in a SINGLE srun so
# the ephemeral /tmp checkpoint is shared between train and eval.
#
# Eval args match eval_wabs_w_gb10.sh exactly (same 8 segments), so contact-W is
# directly comparable to the supervision-only baseline (deployed head W=308mm,
# reanchor16 ceiling=61.5mm on HOT3D P0001).
#
# Run on a gb10 node:
#   srun --account=3dv --gpus=gb10:1 --mem=48G --time=01:30:00 --pty bash run_p4_contact_gb10.sh
#
# NOTE: this is a bounded SIGNAL probe. global_step counts OPTIMIZER steps, so to
# keep the GS-on train inside the srun we force grad_accum=1 (1 micro-batch per
# step) and max_steps=50 -> 50 forward+backward passes, ~10-15 min. It answers
# "does it run end-to-end and does W move toward 61.5". For a real number, drop
# the overrides (restores grad_accum=8 / full run) and eval on a held-out split.
set -uo pipefail

source /work/scratch/dmonopoli/venv_gb10/bin/activate
export XDG_CACHE_HOME=/tmp/xc TORCH_HOME=/tmp/th HF_HOME=/tmp/hf MPLCONFIGDIR=/tmp/mpl
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
mkdir -p /tmp/xc /tmp/th /tmp/hf /tmp/mpl
cd /home/dmonopoli/FF-4DGS-Ego
DR=/work/courses/3dv/team25/data/hot3d_aria/preprocessed_pinhole_f609

echo "######## TRAIN p4_contact (bounded: 50 opt-steps, grad_accum=1) @ $(date) on $(hostname) ########"
python -u -m scripts.train_hand_head --config configs/exp_p4_contact.yaml \
  training.output_dir=/tmp/rt_contact training.max_steps=50 training.grad_accum_steps=1 \
  training.root_anchor_warmup_steps=8 training.kp3d_abs_warmup_steps=8 \
  training.log_every=5 training.val_every=25 training.val_max_batches=8 \
  debug.enabled=true debug.max_sequences=8 2>&1 \
  | grep --line-buffered -vE "Loaded GT joints|Loaded 2D GT|No calibration for|\[VIS\]"

echo "######## build eval config (warm-start the contact-trained head) ########"
python - <<'PY'
import yaml
c = yaml.safe_load(open("configs/exp_p4_contact.yaml"))
c["model"]["warm_start_hand_head"] = "/tmp/rt_contact/hand_head_final.pt"
yaml.safe_dump(c, open("/tmp/eval_contact.yaml", "w"))
print("wrote /tmp/eval_contact.yaml -> warm_start /tmp/rt_contact/hand_head_final.pt")
PY

echo "######## W-EVAL: contact-anchored head (same args as wabs baseline) ########"
python -u -m scripts.eval_world_space --config /tmp/eval_contact.yaml \
  --data_root "$DR" \
  --max_seqs 4 --max_segs 2 --segment_len 128 --clip_len 16 --stride 8 --wa_short 16 \
  --out /tmp/world_eval_contact.json 2>&1 \
  | grep --line-buffered -vE "fwd\+lift|Loaded GT joints|Loaded 2D GT|No calibration for|\[VIS\]"
echo "######## DONE @ $(date) ########"
echo "Compare: contact W-MPJPE pooled vs baseline 308.3, ceiling reanchor16=61.5"
