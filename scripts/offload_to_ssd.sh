#!/bin/bash
# Move this checkout's bulky directories to the external SSD and leave symlinks behind (see evolve/storage.py).
# Safe to rerun: a directory that is already a link is skipped. Each move is copy -> verify -> delete -> link, and
# nothing is deleted unless the copy has the same file count and total bytes. Stop jobs that write there first.
#   scripts/offload_to_ssd.sh            # move
#   DRY=1 scripts/offload_to_ssd.sh      # show what would move
set -euo pipefail
cd "$(dirname "$0")/.."
VOL=${OFFLOAD_VOLUME:-/Volumes/Crucial2TB}
DEST="$VOL/MacOffload/$(basename "$PWD")"
mount | grep -q " on $VOL " || { echo "$VOL is not mounted"; exit 1; }
[ -w "$VOL" ] || { echo "$VOL is not writable (Privacy & Security > Files and Folders > Removable Volumes)"; exit 1; }
DIRS=(cache evolve/runs evolve/rl_plan/out simulator_output rl_models scratch)  # untracked dirs only

stat_dir() { find "$1" -type f | wc -l | tr -d ' '; du -sk "$1" | cut -f1; }
for d in "${DIRS[@]}"; do
  if [ -L "$d" ]; then echo "skip $d (already -> $(readlink "$d"))"; continue; fi
  t="$DEST/$d"
  if [ ! -e "$d" ]; then
    [ -n "${DRY:-}" ] && { echo "would create $t and link $d"; continue; }
    mkdir -p "$t" "$(dirname "$d")" && ln -s "$t" "$d" && echo "created $d -> $t"; continue
  fi
  echo "$d: $(du -sh "$d" | cut -f1) -> $t"
  [ -n "${DRY:-}" ] && continue
  mkdir -p "$t"
  rsync -a "$d/" "$t/"
  a=$(find "$d" -type f | wc -l | tr -d ' '); b=$(find "$t" -type f | wc -l | tr -d ' ')
  sa=$(find "$d" -type f -exec stat -f %z {} + 2>/dev/null | awk '{s+=$1} END {print s+0}')
  sb=$(find "$t" -type f -exec stat -f %z {} + 2>/dev/null | awk '{s+=$1} END {print s+0}')
  if [ "$a" != "$b" ] || [ "$sa" != "$sb" ]; then echo "VERIFY FAILED for $d ($a/$sa vs $b/$sb); original kept"; exit 1; fi
  rm -rf "$d" && ln -s "$t" "$d" && echo "moved $d ($a files, $sa bytes)"
done
df -h "$HOME" | tail -1
