#!/bin/bash
#SBATCH --job-name=db-reclaim
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=/home/michaelvolk/Documents/projects/torchcell.worktrees/exp/033-env-chemgen-pooled/experiments/033-env-chemgen-pooled/slurm/output/%j_%x.out
#SBATCH --error=/home/michaelvolk/Documents/projects/torchcell.worktrees/exp/033-env-chemgen-pooled/experiments/033-env-chemgen-pooled/slurm/output/%j_%x.out
#SBATCH --no-requeue

# Remove regenerable query-build stage directories from /db, one build root per job.
#
# /db reached 95 percent (440 G free of 7.3 T) because finished builds keep their
# upstream stages. `processed` is the live dataset and is never removable here. Two
# kinds of directory are, and each has its own gate:
#
#   STAGES     stages of a build that SUCCEEDED. Removable only if the matching
#              <stage>.tar.zst and its .sha256 sidecar exist on /bulk and the sha256
#              still verifies in this job, immediately before the removal. A stage
#              whose archive is missing, unverifiable, or altered is kept, and the job
#              fails rather than continuing past it.
#   UNARCHIVED directories left by a build that did NOT finish, which were never
#              archived because their contents are a truncated prefix of a stage that
#              was later rebuilt correctly. Removable only if the directory holds an
#              LMDB and NO STAGE_COMPLETE marker, which is the pipeline's own
#              definition of an unusable stage (torchcell/data/neo4j_query_raw.py).
#
# Every removal is appended to /db/deprecated/<date>/MANIFEST.tsv first, matching the
# existing entries there: path, timestamp, size, reason. That manifest is the record;
# the bytes themselves are not moved aside, because a move within /db frees nothing and
# a move to /bulk would copy terabytes that are already archived there.
#
# Usage, via one sbatch per build root:
#   sbatch --export=ALL,BUILD_ROOT=...,ARCHIVE_DIR=...,STAGES="raw conversion" ...
#   sbatch --export=ALL,BUILD_ROOT=...,UNARCHIVED="conversion.partial-2748" ...
set -euo pipefail

BUILD_ROOT=${BUILD_ROOT:?BUILD_ROOT is required}
ARCHIVE_DIR=${ARCHIVE_DIR:-}
STAGES=${STAGES:-}
UNARCHIVED=${UNARCHIVED:-}

MANIFEST_DIR=/db/deprecated/$(date +%Y.%m.%d)
MANIFEST=$MANIFEST_DIR/MANIFEST.tsv
mkdir -p "$MANIFEST_DIR"

echo "Running on $(hostname) at $(date -Is)"
echo "BUILD_ROOT=$BUILD_ROOT"
df -h /db | tail -1

case "$BUILD_ROOT" in
  /db/experiments/*) ;;
  *) echo "ABORT: BUILD_ROOT must be under /db/experiments" >&2; exit 1 ;;
esac

# Record the removal BEFORE it happens, so an interrupted job still leaves a trail.
note() {
  printf '%s\t%s\t%s\t%s\t%s\n' "$1" "$2" "$(date -Is)" "$3" "$4" >> "$MANIFEST"
}

for stage in $STAGES; do
  [ "$stage" != "processed" ] || { echo "ABORT: processed is the live dataset" >&2; exit 1; }
  dir=$BUILD_ROOT/$stage
  [ -d "$dir" ] || { echo "skip $stage: not present"; continue; }
  [ -n "$ARCHIVE_DIR" ] || { echo "ABORT: $stage needs ARCHIVE_DIR" >&2; exit 1; }
  sidecar=$ARCHIVE_DIR/$stage.tar.zst.sha256
  [ -f "$ARCHIVE_DIR/$stage.tar.zst" ] || { echo "ABORT: no archive for $stage" >&2; exit 1; }
  [ -f "$sidecar" ] || { echo "ABORT: no sha256 sidecar for $stage; it cannot be verified" >&2; exit 1; }
  ( cd "$ARCHIVE_DIR" && sha256sum -c "$(basename "$sidecar")" ) \
    || { echo "ABORT: $stage.tar.zst does not match its sha256; keeping $dir" >&2; exit 1; }
  size=$(du -sh "$dir" | cut -f1)
  note "$stage" "$dir" "$size" "regenerable stage of a completed build, archived and sha256-verified at $ARCHIVE_DIR/$stage.tar.zst"
  echo "removing $dir ($size)"
  rm -rf "$dir"
done

for dir_name in $UNARCHIVED; do
  case "$dir_name" in
    processed|raw|conversion|deduplication|aggregation)
      echo "ABORT: $dir_name is a real stage name, not a leftover" >&2; exit 1 ;;
  esac
  dir=$BUILD_ROOT/$dir_name
  [ -d "$dir" ] || { echo "skip $dir_name: not present"; continue; }
  [ ! -e "$dir/STAGE_COMPLETE" ] \
    || { echo "ABORT: $dir carries STAGE_COMPLETE, so it is a finished stage" >&2; exit 1; }
  [ -e "$dir/lmdb/data.mdb" ] \
    || { echo "ABORT: $dir holds no LMDB; not the leftover this job expects" >&2; exit 1; }
  size=$(du -sh "$dir" | cut -f1)
  note "$dir_name" "$dir" "$size" "truncated stage from a cancelled build, no STAGE_COMPLETE, superseded by the rebuilt stage; never archived"
  echo "removing $dir ($size)"
  rm -rf "$dir"
done

echo "=== manifest ==="
cat "$MANIFEST"
df -h /db | tail -1
echo "finished at $(date -Is)"
