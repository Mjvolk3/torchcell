#!/bin/bash
# scripts/backup_mirrors_to_bulk.sh
#
# Weekly, non-destructive copy of the two provenance mirrors on /scratch into the /bulk
# archive tier, so a rebuild never depends on a live URL (sources have vanished before).
#
#   $DATA_ROOT/torchcell-library/   paper PDFs, OCR, SI and released data, per-key manifest.json
#   $DATA_ROOT/torchcell-raw/       raw files a dataset loader consumed for its first successful
#                                   build (the loader records their sha256)
#
# rsync -a without --delete: a file removed on /scratch stays in /bulk (the archive is a
# backstop, never a mirror of deletions). Files that changed on /scratch overwrite the copy;
# sha256-pinned artifacts never change in place, so that only ever refreshes manifests and
# OCR byproducts. Run from cron (scripts/crontab.txt) or by hand:
#
#   bash scripts/backup_mirrors_to_bulk.sh            # copy both mirrors
#   bash scripts/backup_mirrors_to_bulk.sh --dry-run  # itemize what would transfer
#
# Exits non-zero if either rsync fails, so the cron log shows the failure. The one-shot
# migration that seeded /bulk/torchcell-library lives in scripts/migrate_storage_tiers.sh.
set -euo pipefail

SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/projects/torchcell-scratch}"
BULK_ROOT="${BULK_ROOT:-/bulk}"
MIRRORS=(torchcell-library torchcell-raw)

dry=()
if [[ "${1:-}" == "--dry-run" ]]; then
    dry=(--dry-run --itemize-changes)
fi

echo "== backup_mirrors_to_bulk $(date -Is) host=$(hostname)"
status=0
for m in "${MIRRORS[@]}"; do
    src="$SCRATCH_ROOT/$m"
    if [[ ! -d "$src" ]]; then
        echo "MISSING $src" >&2
        status=1
        continue
    fi
    echo "-- $src -> $BULK_ROOT/$m"
    if rsync -a --stats "${dry[@]}" "$src" "$BULK_ROOT/" | grep -E "^(Number of (regular files transferred|files)|Total transferred file size|Total file size)"; then
        :
    else
        echo "rsync FAILED for $m" >&2
        status=1
    fi
done
for m in "${MIRRORS[@]}"; do
    [[ -d "$BULK_ROOT/$m" ]] && du -sh "$BULK_ROOT/$m"
done
echo "== done $(date -Is) status=$status"
exit "$status"
