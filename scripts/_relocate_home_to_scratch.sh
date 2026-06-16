#!/bin/bash
# One-shot: move my big x86 dirs + HF cache off the $HOME quota onto /work/scratch,
# leaving symlinks so all absolute paths keep working. Frees ~6.7G from $HOME.
S=/work/scratch/dmonopoli
echo "START $(date)"
for d in venv_unidepth HaWoR UniDepth; do
  if [ -d "$HOME/$d" ] && [ ! -L "$HOME/$d" ]; then
    echo "rsync $d ..."
    rsync -a "$HOME/$d/" "$S/$d/" && rm -rf "$HOME/$d" && ln -s "$S/$d" "$HOME/$d" && echo "relocated $d"
  else
    echo "skip $d (already symlink or missing)"
  fi
done
if [ -d "$HOME/.cache/huggingface" ] && [ ! -L "$HOME/.cache/huggingface" ]; then
  echo "rsync hf_cache ..."
  rsync -a "$HOME/.cache/huggingface/" "$S/hf_cache/" && rm -rf "$HOME/.cache/huggingface" && \
    ln -s "$S/hf_cache" "$HOME/.cache/huggingface" && echo "relocated hf_cache"
fi
echo "HOME now: $(du -sh "$HOME" 2>/dev/null | cut -f1)"
echo "RELOCATE_DONE $(date)"
