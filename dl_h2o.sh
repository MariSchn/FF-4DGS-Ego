#!/bin/bash
# H2O download via curl (no python -> avoids aarch64 venv on x86 login).
# Creds from $H2O_USER/$H2O_PASS; dest from $H2O_DEST; file subset from $H2O_FILES.
cd "${H2O_DEST:-/work/scratch/dmonopoli/h2o}" || exit 1
BASE="https://h2odataset.ethz.ch/data/dataset"
FILES="${H2O_FILES:-object.zip label_split.zip subject1_ego_v1_1.tar.gz subject2_ego_v1_1.tar.gz subject3_ego_v1_1.tar.gz subject4_ego_v1_1.tar.gz}"
for f in $FILES; do
  echo "=== $f $(date) ==="
  curl -sL -u "$H2O_USER:$H2O_PASS" -C - --retry 5 --retry-delay 10 -o "$f" "$BASE/$f" \
    && echo "OK $f $(du -h "$f" 2>/dev/null | cut -f1)" || echo "FAIL $f"
done
echo "=== H2O BATCH DONE $(date) ==="; du -sh "$(pwd)"
