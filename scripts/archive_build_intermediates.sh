#!/bin/bash
# scripts/archive_build_intermediates.sh <build_root> <dest_dir> <stage> [<stage> ...]
#
# Archive the intermediate stage directories of a query build (raw, conversion,
# deduplication, aggregation) into <dest_dir>/<stage>.tar.zst on cold storage, test
# every archive, and record its sha256 and decompressed byte count. The processed
# directory is the live dataset and is never an argument here.
#
# Measured on the 025 and 029 builds (2026-09-18/19): 4.81 TB of LMDB stage copies
# archived to 27 GB, about 180 to 1, at roughly 17 minutes per 600 GB stage on 16
# threads. The verdict of the test is zstd's exit status; its stdout is the byte
# count, never the word OK.
#
# This script deletes nothing. Removing the stage directories from the build root
# after the archive is verified is the user's action.
#
#   bash scripts/archive_build_intermediates.sh \
#       /db/experiments/030-solid-growth-multi-001-multi-build \
#       /bulk/experiments/030-solid-growth-multi-001-multi-build-intermediates \
#       raw conversion aggregation
set -euo pipefail

if [ $# -lt 3 ]; then
  echo "usage: $0 <build_root> <dest_dir> <stage> [<stage> ...]" >&2
  exit 2
fi
BUILD_ROOT=$1
DEST=$2
shift 2
THREADS=${ZSTD_THREADS:-16}

for stage in "$@"; do
  [ -d "$BUILD_ROOT/$stage" ] || { echo "ABORT: $BUILD_ROOT/$stage is not a directory" >&2; exit 1; }
  [ "$stage" != "processed" ] || { echo "ABORT: processed is the live dataset, not an intermediate" >&2; exit 1; }
done
mkdir -p "$DEST"
LOG="$DEST/archive.log"
echo "archive of $BUILD_ROOT started $(date -Is) on $(hostname)" | tee -a "$LOG"

for stage in "$@"; do
  archive="$DEST/$stage.tar.zst"
  src_bytes=$(du -sb "$BUILD_ROOT/$stage" | cut -f1)
  t0=$(date +%s)
  tar -C "$BUILD_ROOT" -cf - "$stage" | zstd -T"$THREADS" -q -f -o "$archive"
  # zstd -t exits nonzero on any corrupt frame; set -e turns that into an abort.
  # It prints "<file>: N bytes" to stderr, the decompressed size of the tar stream.
  tested=$(zstd -t "$archive" 2>&1)
  dec_bytes=$(echo "$tested" | sed -n 's/.*: *\([0-9][0-9]*\) bytes.*/\1/p' | tail -1)
  [ -n "$dec_bytes" ] || { echo "ABORT: could not read the decompressed size from: $tested" >&2; exit 1; }
  sha=$(sha256sum "$archive" | cut -d' ' -f1)
  echo "$sha  $stage.tar.zst" > "$DEST/$stage.tar.zst.sha256"
  echo "$stage done $(( $(date +%s) - t0 )) s source_bytes=$src_bytes decompressed_bytes=$dec_bytes archive=$(du -h "$archive" | cut -f1) sha256=$sha" | tee -a "$LOG"
done
echo "ALL_DONE $(date -Is)" | tee -a "$LOG"
