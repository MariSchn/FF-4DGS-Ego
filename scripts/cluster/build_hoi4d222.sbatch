#!/bin/bash
#SBATCH --job-name=build222
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=6G
#SBATCH --cpus-per-task=8
#SBATCH --output=/cluster/scratch/dmonopoli/build222_%j.out
# Build the 222-clip HOI4D store that is held out for BOTH us and Hand3R.
#
# The set comes from report/hand3r_fair_eval_clips.txt: their published held-out split, minus every
# clip we trained on, minus every clip sharing an (H,C,S) scene with one we trained on. Our existing
# store covers none of them, which is the point.
#
# FOUR INPUTS, three of them streamed from the HF mirror and one already on disk:
#   video      HOI4D_release.zip      <seq>/align_rgb/image.mp4
#   extrinsics HOI4D_annotations.zip  <seq>/3Dseg/output.log   (2,973 present)
#   handpose   hoi4d_handpose/        Hand_pose/handpose_right_hand/<nested>/*.pickle
#   intrinsics NOT NEEDED: --recover_k solves K by least squares from each pickle's kps2D against
#              its own MANO 3D joints, so camera_params never has to be fetched or aligned.
#
# Compute nodes reach the internet only through eth_proxy, and remotezip streams byte ranges rather
# than pulling the whole 100+GB archive.
set -uo pipefail
S=/cluster/scratch/dmonopoli
cd $S/FF-4DGS-Ego
export PYTHONUNBUFFERED=1
module load eth_proxy 2>/dev/null || true
PY=$S/train_env/bin/python

RAW=$S/hoi4d222_raw
OUT=$S/hoi4d222
LIST=$S/hand3r_fair_eval_clips.txt
mkdir -p "$RAW" "$OUT"

$PY -c "import remotezip, cv2, torch, scipy, smplx, chumpy, decord; print('deps OK')" || exit 1
[ -s "$LIST" ] || { echo "NO CLIP LIST at $LIST"; exit 1; }
echo "clips requested: $(wc -l < "$LIST")"

echo "===== 1. stream video + 3Dseg/output.log for each clip ====="
LIST="$LIST" RAW="$RAW" $PY - <<'PYFETCH'
import os
from remotezip import RemoteZip
REL = "https://huggingface.co/datasets/yinloonga/HOI4D/resolve/main/HOI4D_release.zip"
ANN = "https://huggingface.co/datasets/yinloonga/HOI4D/resolve/main/HOI4D_annotations.zip"
RAW = os.environ["RAW"]
want = [l.strip() for l in open(os.environ["LIST"]) if l.strip()]
print(f"want {len(want)} clips", flush=True)

def nested(flat):                       # ZY..._H1_C12_N26_S200_s01_T2 -> ZY.../H1/C12/N26/S200/s01/T2
    return "/".join(flat.split("_"))

# One RemoteZip session per archive: each open re-reads the central directory, which on a
# 100+GB zip is minutes of latency, so opening per clip would dominate the runtime.
n_vid = n_log = 0
with RemoteZip(REL) as z:
    idx = {}
    for n in z.namelist():
        if n.endswith("align_rgb/image.mp4"):
            idx["_".join(n.split("/")[1:8])] = n
    print(f"mirror indexes {len(idx)} videos", flush=True)
    for i, s in enumerate(want):
        d = os.path.join(RAW, s, "align_rgb"); os.makedirs(d, exist_ok=True)
        dst = os.path.join(d, "image.mp4")
        if os.path.exists(dst) and os.path.getsize(dst) > 100_000:
            n_vid += 1; continue
        if s not in idx:
            print(f"MISS_VIDEO {s}", flush=True); continue
        with open(dst + ".part", "wb") as f:
            f.write(z.read(idx[s]))
        os.rename(dst + ".part", dst); n_vid += 1
        if i % 20 == 0:
            print(f"  video {i+1}/{len(want)} ok={n_vid}", flush=True)
print(f"VIDEOS ok={n_vid}/{len(want)}", flush=True)

with RemoteZip(ANN) as z:
    idx = {}
    for n in z.namelist():
        if n.endswith("3Dseg/output.log"):
            idx["_".join(n.split("/")[1:8])] = n
    print(f"mirror indexes {len(idx)} output.log", flush=True)
    for i, s in enumerate(want):
        d = os.path.join(RAW, s, "3Dseg"); os.makedirs(d, exist_ok=True)
        dst = os.path.join(d, "output.log")
        if os.path.exists(dst) and os.path.getsize(dst) > 100:
            n_log += 1; continue
        if s not in idx:
            print(f"MISS_LOG {s}", flush=True); continue
        with open(dst + ".part", "wb") as f:
            f.write(z.read(idx[s]))
        os.rename(dst + ".part", dst); n_log += 1
        if i % 20 == 0:
            print(f"  log {i+1}/{len(want)} ok={n_log}", flush=True)
print(f"LOGS ok={n_log}/{len(want)}", flush=True)
PYFETCH

echo "===== 2. lay out the tree preprocess_hoi4d expects ====="
# It wants <hoi4d_root>/<seq nested>/3Dseg/output.log and reads RGB from
# <rgb_root>/<seq flat>/images or a video. We give it the nested layout by symlink so nothing is
# copied twice.
$PY - <<PYLAYOUT
import os
RAW="$RAW"
for s in sorted(os.listdir(RAW)):
    src=os.path.join(RAW,s)
    if not os.path.isdir(src): continue
    dst=os.path.join(RAW,"nested","/".join(s.split("_")))
    os.makedirs(os.path.dirname(dst),exist_ok=True)
    if not os.path.lexists(dst): os.symlink(src,dst)
print("nested layout ready")
PYLAYOUT

echo "===== 3. convert each clip ====="
ok=0; fail=0
while read -r s; do
  [ -z "$s" ] && continue
  [ -f "$OUT/$s/hand_data/gt_joints_cache_cam_v2.pt" ] && { ok=$((ok+1)); continue; }
  nested=$(echo "$s" | tr '_' '/')
  $PY -m scripts.preprocessing.preprocess_hoi4d \
      --hoi4d_root "$RAW/nested" \
      --rgb_root "$RAW/nested" \
      --handpose_root "$S/hoi4d_handpose/Hand_pose/handpose_right_hand" \
      --recover_k \
      --seq "$nested" \
      --out "$OUT" \
      --mano_model models/MANO --res 1080 > /dev/null 2>&1 \
    && { ok=$((ok+1)); } || { fail=$((fail+1)); echo "CONVFAIL $s"; }
  [ $(( (ok+fail) % 20 )) -eq 0 ] && echo "  convert ok=$ok fail=$fail"
  # ANTI-TACO GATE. The TACO run failed 2311 times in a row on one wrong keyword because the
  # smoke test had exercised a code path that returned before the write. Here the first five
  # clips ARE the smoke test: if none of them produced a cache, the layout or the arguments are
  # wrong for all 222 and there is nothing to gain from discovering that 217 clips later.
  if [ $((ok+fail)) -eq 5 ] && [ $ok -eq 0 ]; then
    echo "ABORT: first 5 conversions all failed. Re-run one by hand without the /dev/null redirect."
    nested=$(head -1 "$LIST" | tr '_' '/')
    $PY -m scripts.preprocessing.preprocess_hoi4d --hoi4d_root "$RAW/nested" --rgb_root "$RAW/nested" \
        --handpose_root "$S/hoi4d_handpose/Hand_pose/handpose_right_hand" --recover_k \
        --seq "$nested" --out "$OUT" --mano_model models/MANO --res 1080 2>&1 | tail -25
    exit 4
  fi
done < "$LIST"
echo "CONVERT ok=$ok fail=$fail"

echo "===== 4. verify the box store ====="
$PY -m scripts.verify_box_store --data_root "$OUT" 2>&1 | tail -20
echo "sequences written: $(ls "$OUT" 2>/dev/null | wc -l)"
du -sh "$OUT" 2>/dev/null
echo "### BUILD222_DONE ###"
