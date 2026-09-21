#!/bin/bash
#SBATCH --job-name=030-query-build
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=63
#SBATCH --mem=256G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/michaelvolk/Documents/projects/torchcell.worktrees/030-build/experiments/030-solid-growth-multi/slurm/output/%j_%x.out
#SBATCH --error=/home/michaelvolk/Documents/projects/torchcell.worktrees/030-build/experiments/030-solid-growth-multi/slurm/output/%j_%x.out
#SBATCH --no-requeue

# Phase 2 of the 030 build: conversion -> aggregation -> processed + indices + label
# table, from a raw store that scripts/gh_query_build_001_raw.slurm has already written
# and marked complete. The build is two jobs because the query is light and long and the
# aggregation is heavy and shorter; one job sized for the aggregation held 80 CPUs and
# 320 GB through a 19 h query (job 2681, cancelled at 47 min with 1.7 GB resident) and
# kept GPU training off the node.
#
# MEMORY. 029 peaked at 155 GB of resident memory for 26.8M records (sacct -j 2400),
# and 030 returns about 43.8M, so a linear scaling says about 255 GB. 256 GB is half the
# 513 GB node, the ceiling the machine's GPU training needs kept free, and at the 4.1 GB
# MaxMemPerCPU cap it implies 63 CPUs. The projection sits at that ceiling, so an OOM in
# aggregation pass 1 (one key per record) is possible; MEM_GB / CPUS below let a resubmit
# raise it (320 GB needs 80 CPUs) for the hours this job runs when the node is free.
#
# TIME, projected from the measured 029 rates scaled to 43.8M records: conversion about
# 1 h 15, aggregation about 6 h 30, processed copy 30 min, label table and indices 15 to
# 20 h. The index stage is the least certain: it grew fivefold from 025 to 029 because
# without the merge every record carries all its entries, and 030 has both more records
# and more entries per record than 029.
#
# DISK. Four stage copies. 029 took 615 + 614 + 547 + 547 GB for 26.8M records, so 030
# needs about 3.7 TB. The preflight below refuses to start without it, because a build
# that dies on ENOSPC at hour 20 has wasted the node and left a partial store. The 025
# and 029 intermediates were archived and verified under /bulk/experiments/*-intermediates/
# and then deleted by the user (2026-09-21), which took /db from 523 GB free to 4.9 TB.
#
# ARCHIVE AS WE GO. Once processed is written, the tail of this job archives the three
# intermediate stages to /bulk with scripts/archive_build_intermediates.sh (tar + zstd,
# tested by exit status, sha256 recorded). 025 and 029 compressed 4.81 TB to 27 GB, about
# 180 to 1, at about 17 minutes per 600 GB stage. Deleting the stage directories from
# /db after the archive reports ALL_DONE stays the user's action, never this script's.
#
# PRECONDITIONS. raw/STAGE_COMPLETE must exist (checked below): the raw stage is the
# phase-1 job, and a partial raw store is refused, not resumed. DmfKuzmin2018Dataset and
# DmfKuzmin2020Dataset were re-served with the query-strain fitness records of main
# 4c4a4f950 on 2026-09-21 (release 1.2, increments 2677 and 2678), which is what makes
# the trigenic reconstruction recoverable past 0.54.
#
# The job reads its code from a DETACHED worktree pinned at the build commit, so the
# branch worktree can keep moving while the build runs:
#
#   git worktree add --detach ~/Documents/projects/torchcell.worktrees/030-build <sha>
#   cp <branch worktree>/.env ~/Documents/projects/torchcell.worktrees/030-build/.env
#   sbatch ~/Documents/projects/torchcell.worktrees/030-build/experiments/030-solid-growth-multi/scripts/gh_query_build_001.slurm

echo "Running on $(hostname) at $(date)"
source ~/.zshrc
conda activate ~/miniconda3/envs/torchcell

PROJECT_ROOT="${PROJECT_ROOT:-/home/michaelvolk/Documents/projects/torchcell.worktrees/030-build}"
cd "$PROJECT_ROOT" || exit 1
echo "PROJECT_ROOT=$PROJECT_ROOT ($(git -C "$PROJECT_ROOT" rev-parse --short HEAD 2>/dev/null))"
export PYTHONPATH="$PROJECT_ROOT"

set -a; source "$PROJECT_ROOT/.env"; set +a
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="torchcell"

# --- preflight: refuse to start without room for the four stage copies ---
NEED_GB="${NEED_GB:-3800}"
FREE_GB=$(df -BG --output=avail /db | tail -1 | tr -dc '0-9')
echo "/db free ${FREE_GB} GB; this build needs about ${NEED_GB} GB"
if [ "$FREE_GB" -lt "$NEED_GB" ]; then
  echo "ABORT: not enough space on /db." >&2
  echo "  The 025 and 029 build intermediates are archived under /bulk:" >&2
  du -sh /db/experiments/*/raw /db/experiments/*/conversion \
        /db/experiments/*/aggregation /db/experiments/*/deduplication 2>/dev/null >&2
  echo "  Deleting them is the user's action, not this script's." >&2
  exit 1
fi

DB_ROOT=/db/experiments/030-solid-growth-multi-001-multi-build
LINK="$DATA_ROOT/data/torchcell/experiments/030-solid-growth-multi/001-multi-build"
mkdir -p "$DB_ROOT" "$(dirname "$LINK")"
[[ -e "$LINK" ]] || ln -s "$DB_ROOT" "$LINK"
echo "dataset root: $LINK -> $(readlink -f "$LINK")"
df -h /db | tail -1
if [ ! -e "$DB_ROOT/raw/STAGE_COMPLETE" ]; then
  echo "ABORT: $DB_ROOT/raw/STAGE_COMPLETE is missing; run scripts/gh_query_build_001_raw.slurm first." >&2
  exit 1
fi
echo "raw stage complete: $(du -sh "$DB_ROOT/raw" | cut -f1)"

python experiments/030-solid-growth-multi/scripts/query.py || exit 1
echo "build finished at $(date)"
df -h /db | tail -1

# --- archive as we go: the intermediates leave /db for /bulk once processed exists ---
[ -d "$DB_ROOT/processed" ] || { echo "ABORT: no processed directory, nothing to archive behind" >&2; exit 1; }
ARCHIVE_DEST="/bulk/experiments/$(basename "$DB_ROOT")-intermediates"
ZSTD_THREADS=16 bash scripts/archive_build_intermediates.sh "$DB_ROOT" "$ARCHIVE_DEST" raw conversion aggregation
echo "archive finished at $(date); stage directories under $DB_ROOT are yours to delete"
