#!/usr/bin/env bash
# Periodic janitor for the deprecation graveyard (companion to scripts/deprecate.sh).
#
# /deprecate MOVES things into a graveyard and NEVER auto-deletes. This script is the
# retention sweep: it removes ONLY graveyard entries that have STAYED past the retention
# window (default 14 days), keeping anything more recent. Age is read from each entry's
# moved-in timestamp encoded in the graveyard dir name (`YYYY-MM-DD_HHMMSS__<name>`, written
# by deprecate.sh), so "stayed N days" is measured exactly from when it was deprecated --
# independent of mtime or when this sweep happens to run. Running it more often is always
# safe: it can never remove anything younger than the window.
#
# Safety: only touches subdirectories of a graveyard whose path contains
# `torchcell-deprecated`, only entries matching the deprecate timestamp pattern, and refuses
# any graveyard inside $DATA_ROOT (a truncated path there would hit the data root).
#
# Usage: scripts/purge_deprecated.sh [--dry-run]
#   PURGE_AFTER_DAYS=14   retention window in days
#   DEPRECATED_DIRS=a:b   colon-separated graveyards to sweep
#                         (default: /tmp/torchcell-deprecated:/scratch/projects/torchcell-deprecated)
set -euo pipefail

DAYS="${PURGE_AFTER_DAYS:-14}"
DRY_RUN=0
[ "${1:-}" = "--dry-run" ] && DRY_RUN=1

IFS=':' read -r -a GRAVEYARDS \
  <<<"${DEPRECATED_DIRS:-/tmp/torchcell-deprecated:/scratch/projects/torchcell-deprecated}"

now=$(date +%s)
cutoff=$((now - DAYS * 86400))
purged=0
kept=0

for gy in "${GRAVEYARDS[@]}"; do
  [ -d "$gy" ] || continue
  # Only ever sweep a real graveyard, and never one inside the data root.
  case "$gy" in
    *torchcell-deprecated*) : ;;
    *) echo "purge: refusing non-graveyard path: $gy" >&2; continue ;;
  esac
  if [ -n "${DATA_ROOT:-}" ]; then
    case "$gy/" in
      "$DATA_ROOT"/*)
        echo "purge: refusing graveyard inside DATA_ROOT: $gy" >&2
        continue
        ;;
    esac
  fi

  for entry in "$gy"/*/; do
    [ -e "$entry" ] || continue          # no matches -> glob stays literal
    base=$(basename "$entry")
    ts=${base%%__*}                       # expect YYYY-MM-DD_HHMMSS
    d=${ts%_*}
    t=${ts#*_}
    if ! [[ "$d" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ && "$t" =~ ^[0-9]{6}$ ]]; then
      echo "purge: skip (not a deprecate-timestamp entry): $base"
      continue
    fi
    epoch=$(date -d "$d ${t:0:2}:${t:2:2}:${t:4:2}" +%s 2>/dev/null) \
      || { echo "purge: skip (unparseable timestamp): $base"; continue; }
    age_days=$(((now - epoch) / 86400))
    if [ "$epoch" -lt "$cutoff" ]; then
      if [ "$DRY_RUN" = 1 ]; then
        echo "WOULD purge (${age_days}d old): $entry"
      else
        rm -rf "$entry"
        echo "purged (${age_days}d old): $entry"
      fi
      purged=$((purged + 1))
    else
      kept=$((kept + 1))
    fi
  done
done

verb="removed"
[ "$DRY_RUN" = 1 ] && verb="would be removed"
echo "purge: ${purged} ${verb}, ${kept} kept (retention ${DAYS}d, $(date -u +%FT%TZ))."
