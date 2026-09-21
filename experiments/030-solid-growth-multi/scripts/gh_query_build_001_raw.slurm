#!/bin/bash
#SBATCH --job-name=030-query-raw
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/michaelvolk/Documents/projects/torchcell.worktrees/030-build/experiments/030-solid-growth-multi/slurm/output/%j_%x.out
#SBATCH --error=/home/michaelvolk/Documents/projects/torchcell.worktrees/030-build/experiments/030-solid-growth-multi/slurm/output/%j_%x.out
#SBATCH --no-requeue

# Phase 1 of the 030 build: the Neo4j query into raw/lmdb, and nothing else.
#
# The query is single-threaded and light (job 2681 measured 630 records/s at 1.7 GB
# resident after 47 min) but long: about 43.8M records, so roughly 19 h. It gets the
# smallest useful allocation so GPU training keeps the rest of the node; the
# memory-heavy stages run afterwards in scripts/gh_query_build_001.slurm, which refuses
# to start until raw/STAGE_COMPLETE exists.
#
# The job reads its code from the DETACHED worktree pinned at the build commit:
#
#   git worktree add --detach ~/Documents/projects/torchcell.worktrees/030-build <sha>
#   cp <branch worktree>/.env ~/Documents/projects/torchcell.worktrees/030-build/.env
#   sbatch ~/Documents/projects/torchcell.worktrees/030-build/experiments/030-solid-growth-multi/scripts/gh_query_build_001_raw.slurm

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

# raw is one of the four stage copies; 029's raw was 615 GB for 26.8M records
NEED_GB="${NEED_GB:-1200}"
FREE_GB=$(df -BG --output=avail /db | tail -1 | tr -dc '0-9')
echo "/db free ${FREE_GB} GB; the raw stage needs about ${NEED_GB} GB"
if [ "$FREE_GB" -lt "$NEED_GB" ]; then
  echo "ABORT: not enough space on /db." >&2
  exit 1
fi

DB_ROOT=/db/experiments/030-solid-growth-multi-001-multi-build
LINK="$DATA_ROOT/data/torchcell/experiments/030-solid-growth-multi/001-multi-build"
mkdir -p "$DB_ROOT" "$(dirname "$LINK")"
[[ -e "$LINK" ]] || ln -s "$DB_ROOT" "$LINK"
echo "dataset root: $LINK -> $(readlink -f "$LINK")"
if [ -e "$DB_ROOT/raw/lmdb/data.mdb" ] && [ ! -e "$DB_ROOT/raw/STAGE_COMPLETE" ]; then
  echo "ABORT: $DB_ROOT/raw holds a partial raw store (no STAGE_COMPLETE); move it aside first." >&2
  exit 1
fi

python experiments/030-solid-growth-multi/scripts/query.py --stage raw || exit 1
ls -la "$DB_ROOT/raw"
du -sh "$DB_ROOT/raw"
echo "raw stage finished at $(date); submit scripts/gh_query_build_001.slurm for the remaining stages"
