#!/bin/bash
# Download a spread of HOI4D sequences (one per object category) from the
# Livioni HF mirror, extracting ONLY RGB (images/) + dense GT depth (raw_depth/)
# and deleting the tar — to stay within the scratch quota. For the dense-depth
# experiment (RGB + metric depth; no hand/pose annotations needed).
set -uo pipefail
cd /work/scratch/dmonopoli/hoi4d
SEQS=(
  ZY20210800001_H1_C1_N19_S100_s02_T1
  ZY20210800001_H1_C2_N11_S212_s03_T2
  ZY20210800001_H1_C3_N01_S54_s05_T2
  ZY20210800001_H1_C4_N01_S19_s05_T1
  ZY20210800001_H1_C5_N15_S57_s03_T1
  ZY20210800001_H1_C6_N01_S247_s01_T1
  ZY20210800001_H1_C7_N11_S280_s03_T5
)
BASE="https://huggingface.co/datasets/Livioni/hoi4d/resolve/main"
for s in "${SEQS[@]}"; do
  if [ -d "$s/images" ]; then echo "skip $s (exists)"; continue; fi
  echo "=== downloading $s ==="
  curl -sL --fail -o "$s.tar.gz" "$BASE/$s.tar.gz" || { echo "DL FAIL $s"; rm -f "$s.tar.gz"; continue; }
  tar xzf "$s.tar.gz" "$s/images" "$s/raw_depth" 2>/dev/null \
    || { echo "partial-extract failed, full extract"; tar xzf "$s.tar.gz" 2>/dev/null; }
  rm -f "$s.tar.gz"
  echo "OK $s: imgs=$(ls $s/images 2>/dev/null | wc -l) depth=$(ls $s/raw_depth 2>/dev/null | wc -l)"
done
echo "=== BATCH DONE $(date) ==="
df -h /work/scratch/dmonopoli 2>/dev/null | tail -1
du -sh /work/scratch/dmonopoli/hoi4d 2>/dev/null
