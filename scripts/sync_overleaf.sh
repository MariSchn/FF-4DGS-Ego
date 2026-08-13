#!/usr/bin/env bash
# Push the local paper into the shared Overleaf project without orphaning Cyrus's comments.
#
# WHY THIS IS NOT `cp -r` FOLLOWED BY `git push`. The Overleaf project is where the supervisor
# leaves review comments, and an Overleaf comment is anchored to a text range inside a file. A
# push that rewrites a line detaches every comment anchored to it, silently, with no way to get
# them back. So this script never pushes blind: it prints, per file, exactly what would change on
# Overleaf's side, and only writes when told to.
#
# It also refuses to upload build artefacts. Overleaf compiles the project itself, so pushing
# main.pdf/aux/log/bbl either does nothing or fights the remote compiler, and the working-copy
# leftovers (.pregate, .prelw, .DS_Store) have no business in a shared project.
#
#   1. Get a token: Overleaf > Account Settings > Git Integration > "Create token".
#      ETH members have Overleaf Professional, which includes the git bridge.
#   2. Dry run, which is the default and never writes anything:
#        OVERLEAF_TOKEN=olp_xxx scripts/sync_overleaf.sh
#   3. Read the diff. If a hunk touches a paragraph Cyrus commented on, expect that comment to
#      detach, and consider editing that paragraph in the Overleaf web editor instead.
#   4. Push:
#        OVERLEAF_TOKEN=olp_xxx scripts/sync_overleaf.sh --push
#
# Pulling is automatic and happens first, so edits Cyrus made in the browser are merged into the
# clone before ours land on top.
set -euo pipefail

PROJECT_ID="${OVERLEAF_PROJECT_ID:-6a69be93f785df27754e492e}"
PAPER_DIR="${PAPER_DIR:-$HOME/Downloads/WorldHand4DGS_ICLR27}"
CLONE_DIR="${OVERLEAF_CLONE:-$HOME/git/3dv/overleaf-worldhand4dgs}"
PUSH=0
REFLOW=0
CLOBBER=0
CONFLICTS=0
for a in "$@"; do
  case "$a" in
    --push) PUSH=1 ;;
    # Push files that differ only in line wrapping too. Costs the Overleaf comment anchors on
    # those lines and gains nothing in the compiled document, so it is opt-in.
    --include-reflow) REFLOW=1 ;;
    # Overwrite files that were last edited in the Overleaf web editor. Only correct when their
    # edit has already been merged into the local copy by hand.
    --clobber-remote-edits) CLOBBER=1 ;;
  esac
done

: "${OVERLEAF_TOKEN:?set OVERLEAF_TOKEN (Overleaf > Account Settings > Git Integration)}"
[ -d "$PAPER_DIR" ] || { echo "no paper dir at $PAPER_DIR"; exit 1; }

REMOTE="https://git:${OVERLEAF_TOKEN}@git.overleaf.com/${PROJECT_ID}"

# The token must never reach the terminal, a log, or .git/config. A URL-embedded credential is
# stored verbatim by `git clone`, so the remote is rewritten to the bare URL straight after and
# the credential is supplied per-command instead.
if [ -d "$CLONE_DIR/.git" ]; then
  BR="$(git -C "$CLONE_DIR" symbolic-ref --short HEAD)"
  git -C "$CLONE_DIR" fetch --quiet "$REMOTE" "$BR" 2>&1 | sed "s/${OVERLEAF_TOKEN}/***/g"
  git -C "$CLONE_DIR" merge --ff-only FETCH_HEAD 2>&1 | sed "s/${OVERLEAF_TOKEN}/***/g"
else
  mkdir -p "$(dirname "$CLONE_DIR")"
  git clone --quiet "$REMOTE" "$CLONE_DIR" 2>&1 | sed "s/${OVERLEAF_TOKEN}/***/g"
  git -C "$CLONE_DIR" remote set-url origin "https://git.overleaf.com/${PROJECT_ID}"
fi

# Source only. Everything Overleaf regenerates, and every local scratch file, stays here.
# Only the figures the document actually \includegraphics. The figures/ directory is a workshop:
# it holds Excalidraw sources, standalone TikZ wrappers, .bak/.predepth/.prevignette snapshots, a
# helper .py, and PNG previews of the PDFs. None of that compiles on Overleaf and all of it clutters
# a project a supervisor has to read. Derived from `grep -o '\\includegraphics{...}' main.tex
# Sections/*.tex`; re-derive it when a new figure is added.
FIGURES=(architecture.pdf fig_box_degradation.pdf fig_depth_transfer.pdf qualitative.pdf)
SYNC_GLOBS=(main.tex "Sections/*.tex" "*.bib")
for f in "${FIGURES[@]}"; do SYNC_GLOBS+=("figures/$f"); done
# `pdf` is deliberately NOT in this list. It was, and it silently dropped every figure PDF the
# document \includegraphics, which would have left Overleaf unable to compile. main.pdf is excluded
# by name instead, because it is the only PDF here that is build output rather than an input.
SKIP_RE='\.(aux|log|out|bbl|blg|synctex\.gz|fdb_latexmk|fls)$|^main\.pdf$|\.DS_Store$|\.(pregate|prelw)$'

# True when the newest commit touching this file on the remote is NOT one of ours. Overleaf's git
# bridge attributes web-editor edits to the person who made them, and every commit this script
# makes says "sync from local working copy", so anything else means the supervisor edited that file
# in the browser and we are about to throw his work away.
#
# THIS EXISTS BECAUSE IT HAPPENED. On 2026-08-10 Cyrus rewrote Sections/3method.tex and wrote
# Sections/1intro.tex from scratch in the Overleaf editor. The next push copied our local copies
# straight over both. The `merge --ff-only` above does NOT protect against this: it makes the clone
# current, and then the copy loop overwrites the merged files without looking. Recovery needed
# `git show <his-commit>:<path>` and it was luck that his commit was still in the log.
remote_edited() {
  local rel="$1" last
  last="$(git -C "$CLONE_DIR" log -1 --format='%s' -- "$rel" 2>/dev/null || true)"
  [ -n "$last" ] && [ "$last" != "sync from local working copy" ]
}

# True when two files carry the same characters and differ only in how they are wrapped. Applied
# to .tex and .bib only; for anything else a byte difference is a real difference.
whitespace_only() {
  case "$1" in *.tex|*.bib) ;; *) return 1 ;; esac
  python3 - "$1" "$2" <<'PY'
import re, sys
n = lambda p: re.sub(r'\s+', ' ', open(p, encoding='utf-8', errors='replace').read()).strip()
sys.exit(0 if n(sys.argv[1]) == n(sys.argv[2]) else 1)
PY
}

CHANGED=0
echo "=== what would change ON OVERLEAF (left = Overleaf, right = local) ==="
for g in "${SYNC_GLOBS[@]}"; do
  for src in "$PAPER_DIR"/$g; do
    [ -f "$src" ] || continue
    rel="${src#$PAPER_DIR/}"
    echo "$rel" | grep -qE "$SKIP_RE" && continue
    dst="$CLONE_DIR/$rel"
    if [ ! -f "$dst" ]; then
      echo "--- NEW: $rel"; CHANGED=$((CHANGED+1)); continue
    fi
    if ! cmp -s "$src" "$dst"; then
      # A file that differs ONLY in line wrapping must never be pushed. The remote text would be
      # rewritten line for line, every Overleaf comment anchored to those lines would detach, and
      # not one word of the document would change. This is not hypothetical: on 2026-08-10 the
      # entire Sections/3method.tex diff was our local 100-column reflow of text that was already
      # on Overleaf, byte-identical once whitespace is collapsed.
      if remote_edited "$rel" && [ "$CLOBBER" -eq 0 ]; then
        echo "!!! CONFLICT: $rel was last changed ON OVERLEAF, not by this script."
        echo "    last remote commit: $(git -C "$CLONE_DIR" log -1 --format='%an, %ad, %s' --date=short -- "$rel")"
        echo "    Pushing would discard that edit. Merge it into $PAPER_DIR/$rel first,"
        echo "    or pass --clobber-remote-edits if you are certain."
        CONFLICTS=$((CONFLICTS+1)); continue
      fi
      if [ "$REFLOW" -eq 0 ] && whitespace_only "$dst" "$src"; then
        echo "--- SKIP (whitespace-only reflow, would detach comments for nothing): $rel"
        continue
      fi
      echo "--- MODIFIED: $rel"
      case "$rel" in
        # `head` closes the pipe, diff dies of SIGPIPE, and under `set -o pipefail` that kills the
        # whole script AFTER it has printed a convincing report and BEFORE it copies anything. It
        # looked exactly like a successful dry run that pushed nothing.
        *.tex|*.bib) { diff -u "$dst" "$src" || true; } | head -60 || true ;;
        *) echo "    (binary, $(wc -c <"$dst") -> $(wc -c <"$src") bytes)" ;;
      esac
      CHANGED=$((CHANGED+1))
    fi
  done
done

# A file present on Overleaf and absent locally is almost always something Cyrus added, not
# something we deleted. This script never deletes on the remote; it only reports.
echo "=== on Overleaf but not local (NOT touched by this script) ==="
(cd "$CLONE_DIR" && git ls-files) | while read -r rel; do
  [ -f "$PAPER_DIR/$rel" ] || echo "    $rel"
done

[ "$CHANGED" -eq 0 ] && { echo "nothing to sync"; exit 0; }
echo "=== $CHANGED file(s) differ ==="

if [ "$CONFLICTS" -gt 0 ]; then
  echo "REFUSING TO PUSH: $CONFLICTS file(s) were edited on Overleaf and would be overwritten."
  exit 2
fi

if [ "$PUSH" -ne 1 ]; then
  echo "dry run. re-run with --push to write these to Overleaf."
  exit 0
fi

for g in "${SYNC_GLOBS[@]}"; do
  for src in "$PAPER_DIR"/$g; do
    [ -f "$src" ] || continue
    rel="${src#$PAPER_DIR/}"
    echo "$rel" | grep -qE "$SKIP_RE" && continue
    # Same rule as the report above, or the push would silently write what the dry run said it
    # would skip.
    if [ "$CLOBBER" -eq 0 ] && remote_edited "$rel"; then
      continue
    fi
    if [ "$REFLOW" -eq 0 ] && [ -f "$CLONE_DIR/$rel" ] && whitespace_only "$CLONE_DIR/$rel" "$src"; then
      continue
    fi
    mkdir -p "$(dirname "$CLONE_DIR/$rel")"
    cp "$src" "$CLONE_DIR/$rel"
  done
done

git -C "$CLONE_DIR" add -A
git -C "$CLONE_DIR" -c user.name="Dario Monopoli" -c user.email="dario@ottwittwerstudio.com" \
    commit -q -m "sync from local working copy" || { echo "nothing staged"; exit 0; }
BR="$(git -C "$CLONE_DIR" symbolic-ref --short HEAD)"
git -C "$CLONE_DIR" push --quiet "$REMOTE" "HEAD:$BR" 2>&1 | sed "s/${OVERLEAF_TOKEN}/***/g"
echo "pushed. open https://www.overleaf.com/project/${PROJECT_ID} and recompile."
