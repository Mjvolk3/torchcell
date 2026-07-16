#!/usr/bin/env bash
# Move a path into the deprecation graveyard instead of deleting it (never rm -rf).
#
# Deprecating (not deleting) old datasets, LMDB builds, schemas, or code structures keeps a
# recoverable trail: things filter into a graveyard you can inspect in the VS Code explorer,
# then purge by hand on your own schedule. Portable across machines -- the graveyard defaults
# to /tmp/torchcell-deprecated (a GilaHyper path also shown in the workspace) and is
# overridable with $DEPRECATED_DIR for a same-filesystem move of large /scratch data.
#
# SAFETY: the graveyard must NOT live inside $DATA_ROOT. Use the sibling
# /scratch/projects/torchcell-deprecated, never $DATA_ROOT/torchcell-deprecated -- nesting it
# in the data root means a later hand `rm -rf` that drops the final path component wipes the
# whole data root. This script refuses a $DEPRECATED_DIR inside $DATA_ROOT.
#
# Usage: scripts/deprecate.sh <path> [reason]
#   DEPRECATED_DIR=/scratch/projects/torchcell-deprecated scripts/deprecate.sh <path> "why"
set -euo pipefail

GRAVEYARD="${DEPRECATED_DIR:-/tmp/torchcell-deprecated}"
target="${1:?usage: deprecate.sh <path> [reason]}"
reason="${2:-}"

# Refuse a graveyard nested inside $DATA_ROOT (a truncated rm -rf of it would hit the data).
if [ -n "${DATA_ROOT:-}" ]; then
  case "$GRAVEYARD/" in
    "$DATA_ROOT"/*)
      echo "deprecate: refusing graveyard inside DATA_ROOT ($GRAVEYARD); use the sibling" \
           "/scratch/projects/torchcell-deprecated instead." >&2
      exit 2
      ;;
  esac
fi

if [ ! -e "$target" ]; then
  echo "deprecate: no such path: $target" >&2
  exit 1
fi

# Absolute source path (resolve before the move).
abs="$(cd "$(dirname "$target")" && pwd)/$(basename "$target")"
ts="$(date +%Y-%m-%d_%H%M%S)"
dest="$GRAVEYARD/${ts}__$(basename "$target")"
mkdir -p "$dest"

git_head="$(git -C "$(dirname "$abs")" rev-parse HEAD 2>/dev/null || echo 'not-a-git-repo')"
size="$(du -sh "$abs" 2>/dev/null | cut -f1 || echo '?')"
{
  echo "original_path: $abs"
  echo "deprecated_at: $ts"
  echo "host: $(hostname)"
  echo "user: $(whoami)"
  echo "git_head: $git_head"
  echo "size: $size"
  echo "reason: $reason"
} > "$dest/DEPRECATION.txt"

mv "$target" "$dest/"
echo "deprecated -> $dest/$(basename "$abs")  (${size})"
echo "  manifest: $dest/DEPRECATION.txt"
echo "  graveyard: $GRAVEYARD  (purge by hand; nothing is auto-deleted)"
