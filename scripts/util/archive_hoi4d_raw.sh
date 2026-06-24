#!/bin/bash
# archive_hoi4d_raw.sh — collapse raw HOI4D per-frame dirs into ONE verified zip each.
#
# The cluster limit is INODES (file count), not bytes (per Cyrus). HOI4D burns inodes:
# every seq has ~300 images/*.png + ~300 handpose <frame>.pickle = ~600 tiny files.
# This zips each seq's images/ and its handpose dir into a single archive on scratch,
# VERIFIES the archive (CRC integrity + entry-count match), and ONLY THEN removes the
# originals. The zip stays on scratch, so the seq is always recoverable:
#     (cd $HOI4D/<seq>   && unzip $ARCHIVE/<seq>.images.zip)
#     (cd $HANDPOSE      && unzip $ARCHIVE/<seq>.handpose.zip)
# Re-preprocessing then works exactly as before. Net: ~600 inodes -> 2 per seq.
#
# SAFETY:
#   * DRY-RUN by default. Pass --apply to actually zip + delete.
#   * The 11 ACTIVE C-abs A/B seqs are hard-excluded; the script refuses to touch them
#     even if named explicitly (we still re-preprocess them).
#   * Originals are deleted ONLY after the zip passes both integrity and file-count checks.
#   * Idempotent: a seq whose .zip already exists is skipped.
#
# Usage (on the cluster login node — no GPU needed):
#   bash scripts/util/archive_hoi4d_raw.sh                  # dry-run over all non-active seqs
#   bash scripts/util/archive_hoi4d_raw.sh --apply          # do it
#   bash scripts/util/archive_hoi4d_raw.sh --seqs "S1 S2"   # only these (still excludes active)
set -uo pipefail

HOI4D=${HOI4D:-/work/scratch/dmonopoli/hoi4d}
HANDPOSE=${HANDPOSE:-/work/scratch/dmonopoli/hoi4d_handpose_gt/Hand_pose/handpose_right_hand}
ARCHIVE=${ARCHIVE:-/work/scratch/dmonopoli/hoi4d_archive}
APPLY=0
SEQS_ARG=""

# The 11 sequences in the active camera-frame contact-anchor A/B. NEVER archive these.
EXCLUDE=(
  ZY20210800001_H1_C1_N19_S100_s02_T1
  ZY20210800001_H1_C12_N11_S169_s03_T1
  ZY20210800001_H1_C12_N26_S165_s01_T1
  ZY20210800001_H1_C13_N11_S132_s03_T1
  ZY20210800001_H1_C13_N12_S132_s03_T1
  ZY20210800001_H1_C14_N15_S158_s02_T1
  ZY20210800001_H1_C14_N17_S158_s02_T1
  ZY20210800001_H1_C2_N11_S212_s03_T2
  ZY20210800001_H1_C3_N01_S54_s05_T2
  ZY20210800001_H1_C5_N15_S57_s03_T1
  ZY20210800001_H1_C7_N11_S280_s03_T5
)

while [ $# -gt 0 ]; do
  case "$1" in
    --apply) APPLY=1 ;;
    --seqs)  SEQS_ARG="${2:-}"; shift ;;
    *) echo "unknown arg: $1"; exit 2 ;;
  esac
  shift
done

is_excluded() { local s="$1" e; for e in "${EXCLUDE[@]}"; do [ "$s" = "$e" ] && return 0; done; return 1; }

# count files (not dirs) under a path
nfiles() { [ -d "$1" ] && find "$1" -type f | wc -l | tr -d ' ' || echo 0; }

# zip <parent_dir> <child> <out.zip>  — stores paths relative to parent (clean restore)
make_zip() {
  local parent="$1" child="$2" out="$3"
  ( cd "$parent" && zip -q -r -1 -y "$out.part" "$child" ) || { rm -f "$out.part"; return 1; }
  unzip -tqq "$out.part" >/dev/null 2>&1 || { echo "    integrity FAIL"; rm -f "$out.part"; return 1; }
  local nzip nsrc
  nzip=$(unzip -Z1 "$out.part" | grep -vc '/$')
  nsrc=$(nfiles "$parent/$child")
  if [ "$nzip" -ne "$nsrc" ]; then
    echo "    count FAIL (zip=$nzip src=$nsrc)"; rm -f "$out.part"; return 1
  fi
  mv "$out.part" "$out"
  return 0
}

mkdir -p "$ARCHIVE"

if [ -n "$SEQS_ARG" ]; then
  read -ra SEQS <<< "$SEQS_ARG"
else
  SEQS=()
  for d in "$HOI4D"/*/; do
    [ -d "${d}images" ] || continue
    SEQS+=("$(basename "$d")")
  done
fi

[ "$APPLY" -eq 0 ] && echo "### DRY-RUN (pass --apply to archive+delete) ###"
echo "HOI4D=$HOI4D  HANDPOSE=$HANDPOSE  ARCHIVE=$ARCHIVE"
freed=0; done_n=0
for s in "${SEQS[@]}"; do
  if is_excluded "$s"; then echo "[skip-active] $s"; continue; fi
  img="$HOI4D/$s/images"
  hp="$HANDPOSE/${s//_//}"
  nimg=$(nfiles "$img"); nhp=$(nfiles "$hp"); nsrc=$((nimg + nhp))
  [ "$nsrc" -eq 0 ] && { echo "[skip-empty] $s"; continue; }

  if [ "$APPLY" -eq 0 ]; then
    echo "[would-archive] $s : $nimg img + $nhp handpose = $nsrc files -> 2 zips (frees ~$((nsrc - 2)) inodes)"
    freed=$((freed + nsrc - 2)); continue
  fi

  ok=1
  if [ -d "$img" ]; then
    z="$ARCHIVE/$s.images.zip"
    if [ -f "$z" ]; then echo "[have] $s images.zip"; else
      make_zip "$HOI4D/$s" images "$z" && echo "[zip] $s images ($nimg)" || { echo "[FAIL] $s images — keeping originals"; ok=0; }
    fi
    [ "$ok" -eq 1 ] && [ -f "$z" ] && rm -rf "$img"
  fi
  if [ "$ok" -eq 1 ] && [ -d "$hp" ]; then
    z="$ARCHIVE/$s.handpose.zip"
    if [ -f "$z" ]; then echo "[have] $s handpose.zip"; else
      make_zip "$HANDPOSE" "${s//_//}" "$z" && echo "[zip] $s handpose ($nhp)" || { echo "[FAIL] $s handpose — keeping originals"; ok=0; }
    fi
    [ "$ok" -eq 1 ] && [ -f "$z" ] && rm -rf "$hp"
  fi
  [ "$ok" -eq 1 ] && { freed=$((freed + nsrc - 2)); done_n=$((done_n + 1)); echo "[done] $s (freed ~$((nsrc - 2)) inodes)"; }
done
echo "=== seqs archived: $done_n | inodes freed (est): $freed ==="
[ "$APPLY" -eq 0 ] && echo "### nothing deleted (dry-run) ###"
