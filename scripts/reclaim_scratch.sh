#!/bin/bash
# Free Euler scratch so TACO's 32-frame feature cache fits. DRY RUN unless --go is passed.
#
# Scratch sits at 2.19 TB of a 2.70 TB hard quota. TACO's cache is the last one missing from the
# five-dataset pool and, at the measured ~32 MB per clip, plausibly needs 0.8 to 1.1 TB.
#
# Everything below is either a raw archive whose conversion has been verified, or an artefact of a
# finished experiment whose results already live in a JSON. Nothing here is a headline checkpoint:
# the two that matter are mirrored on HuggingFace (see the hf-checkpoint-backup note).
#
#   bash scripts/reclaim_scratch.sh          # show what would go
#   bash scripts/reclaim_scratch.sh --go     # actually delete
set -uo pipefail
S=/cluster/scratch/dmonopoli
GO=0; [ "${1:-}" = "--go" ] && GO=1
[ $GO -eq 1 ] || echo "=== DRY RUN. Pass --go to delete. ==="

total=0
show () {  # path label
  [ -e "$1" ] || return 0
  local kb; kb=$(du -sk "$1" 2>/dev/null | cut -f1); local gb=$((kb/1048576))
  total=$((total+kb))
  printf '  %-46s %5d GB   %s\n' "$(basename "$1")" "$gb" "$2"
  [ $GO -eq 1 ] && rm -rf "$1"
  return 0
}

echo "RAW ARCHIVES whose conversion is verified"
# taco_ours holds 2311/2311 converted sequences and passed verify_box_store on 2026-08-12.
show "$S/taco_v1"      "converted -> taco_ours (2311/2311, box store verified)"
# hot3d_pinhole_f609 holds 198 undistorted sequences built from this.
show "$S/hot3d_aria"   "converted -> hot3d_pinhole_f609 (198 seqs)"

echo
echo "FINISHED EXPERIMENT ARTEFACTS"
for s in "$S"/step*_snapshot; do
  # The variable-length sweep is complete and its numbers live in results/step*_{fixed32,rand2_32}*.json.
  show "$s" "varlen sweep snapshot, results already in JSON"
done

echo
echo "BROKEN VENVS (scratch purge killed every one on 2026-08-13)"
for e in train_env hawor_env dl_env venv_haptic venv_haptic2 venv_hawor; do
  show "$S/$e" "interpreter does not start; rebuild from the login node when needed"
done

echo
echo "REDUNDANT CHECKPOINTS"
echo "  Keeping one file per run. A run directory is NEVER emptied: if it holds no"
echo "  best_mpjpe.pt, nothing in it is touched, because that run's only artefact"
echo "  may be its final or best-val file."
kept=0; skipped=0
for d in "$S"/checkpoints/*/; do
  [ -d "$d" ] || continue
  if [ ! -f "$d/best_mpjpe.pt" ]; then
    printf '  %-46s SKIPPED, no best_mpjpe.pt to keep\n' "$(basename "$d")"
    skipped=$((skipped+1)); continue
  fi
  for f in "$d"checkpoint_*.pt "$d"hand_head_final.pt "$d"best_val_loss.pt; do
    [ -f "$f" ] || continue
    kb=$(du -sk "$f" 2>/dev/null | cut -f1); total=$((total+kb))
    [ $GO -eq 1 ] && rm -f "$f"
  done
  kept=$((kept+1))
done
echo "  pruned $kept run dirs, skipped $skipped"

echo
printf 'TOTAL RECLAIMED: %d GB\n' $((total/1048576))
[ $GO -eq 1 ] && { echo; lquota 2>/dev/null | grep scratch; }
exit 0
