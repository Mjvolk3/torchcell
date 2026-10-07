#!/bin/bash
#SBATCH --job-name=033-archive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/home/michaelvolk/Documents/projects/torchcell.worktrees/exp/033-env-chemgen-pooled/experiments/033-env-chemgen-pooled/slurm/output/%j_%x.out
#SBATCH --error=/home/michaelvolk/Documents/projects/torchcell.worktrees/exp/033-env-chemgen-pooled/experiments/033-env-chemgen-pooled/slurm/output/%j_%x.out
#SBATCH --no-requeue

# Archive the two regenerable stages of build 001 (raw 108 G, aggregation 106 G on /db)
# to /bulk as tar.zst, test each archive by decompressing it in full, and record its
# sha256 and decompressed byte count. `processed` is the live dataset and is never
# archived here. Build 001 finished as slurm 2929 on 2026-09-28 and wrote nothing
# compressed, so these two stages exist only as uncompressed LMDB on /db.
#
# Precedent: the 025, 029 and 030 builds archived the same way
# (/bulk/experiments/<build>-intermediates/), compressing LMDB stage copies about 180
# to 1 at roughly 1,400 s per 1 TB stage on 16 threads including the test.
#
# The test counts the decompressed bytes itself rather than parsing `zstd -t` output,
# because zstd 1.5.5 prints a TB-scale size only in GiB and the parse aborted build
# 2791's archive after the test had already passed.
#
# This job deletes nothing. Removing the stage directories from /db after archive.log
# says ALL_DONE is the user's action.
set -euo pipefail

SRC=/db/experiments/033-env-chemgen-pooled-001-pooled-build
DST=/bulk/experiments/033-env-chemgen-pooled-001-pooled-build-intermediates
THREADS=16

echo "Running on $(hostname) at $(date -Is)"
df -h /db /bulk | tail -2

# Every archive already on /bulk carries a sha256 sidecar recorded when it was written.
# Re-checking them here is what makes deleting the corresponding /db stages defensible:
# a mismatch means the only copy of that stage is the one still on /db.
echo "=== verifying the archives already on /bulk ==="
for d in /bulk/experiments/*-intermediates; do
  [ -n "$(ls "$d"/*.tar.zst.sha256 2>/dev/null)" ] || { echo "$d: no sha256 sidecar, not verified"; continue; }
  ( cd "$d" && sha256sum -c ./*.tar.zst.sha256 ) | sed "s|^|$(basename "$d") |"
done

echo "=== archiving 033 build 001 ==="
mkdir -p "$DST"
LOG="$DST/archive.log"
echo "archive of $SRC started $(date -Is) on $(hostname)" | tee -a "$LOG"

for stage in raw aggregation; do
  [ -d "$SRC/$stage" ] || { echo "ABORT: $SRC/$stage is not a directory" >&2; exit 1; }
  archive="$DST/$stage.tar.zst"
  src_bytes=$(du -sb "$SRC/$stage" | cut -f1)
  t0=$SECONDS
  tar -C "$SRC" -cf - "$stage" | zstd -T"$THREADS" -q -f -o "$archive"
  dec_bytes=$(zstd -dc -q "$archive" | wc -c; exit "${PIPESTATUS[0]}") \
    || { echo "ABORT: zstd reported a corrupt frame in $archive" >&2; exit 1; }
  [ "$dec_bytes" -gt 0 ] || { echo "ABORT: no byte count from the test of $archive" >&2; exit 1; }
  sha=$(sha256sum "$archive" | cut -d' ' -f1)
  echo "$sha  $stage.tar.zst" > "$archive.sha256"
  echo "$stage done $((SECONDS - t0)) s source_bytes=$src_bytes decompressed_bytes=$dec_bytes archive=$(du -h "$archive" | cut -f1) sha256=$sha" | tee -a "$LOG"
done

echo "ALL_DONE $(date -Is)" | tee -a "$LOG"
df -h /db /bulk | tail -2
echo "finished at $(date -Is)"
