#!/bin/bash
# HOI4D scale-up batch 2: 10 more diverse sequences (RGB + dense depth only).
set -uo pipefail
cd /work/scratch/dmonopoli/hoi4d
SEQS=(
  ZY20210800001_H1_C11_N08_S185_s02_T2
  ZY20210800001_H1_C11_N09_S185_s03_T2
  ZY20210800001_H1_C12_N11_S169_s03_T1
  ZY20210800001_H1_C12_N26_S165_s01_T1
  ZY20210800001_H1_C13_N11_S132_s03_T1
  ZY20210800001_H1_C13_N12_S132_s03_T1
  ZY20210800001_H1_C14_N15_S158_s02_T1
  ZY20210800001_H1_C14_N17_S158_s02_T1
  ZY20210800001_H1_C17_N13_S159_s02_T3
  ZY20210800001_H1_C17_N15_S159_s02_T3
)
BASE="https://huggingface.co/datasets/Livioni/hoi4d/resolve/main"
for s in "${SEQS[@]}"; do
  if [ -d "$s/images" ]; then echo "skip $s (exists)"; continue; fi
  echo "=== downloading $s ==="
  curl -sL --fail -o "$s.tar.gz" "$BASE/$s.tar.gz" || { echo "DL FAIL $s"; rm -f "$s.tar.gz"; continue; }
  tar xzf "$s.tar.gz" "$s/images" "$s/raw_depth" 2>/dev/null \
    || { echo "partial-extract, full"; tar xzf "$s.tar.gz" 2>/dev/null; }
  rm -f "$s.tar.gz"
  echo "OK $s: imgs=$(ls $s/images 2>/dev/null | wc -l) depth=$(ls $s/raw_depth 2>/dev/null | wc -l)"
done
echo "=== BATCH2 DONE $(date) ==="; du -sh /work/scratch/dmonopoli/hoi4d 2>/dev/null
echo "inodes: $(find /work/scratch/dmonopoli -xdev 2>/dev/null | wc -l)"
