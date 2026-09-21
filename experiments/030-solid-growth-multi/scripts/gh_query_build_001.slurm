#!/bin/bash
#SBATCH --job-name=030-query-build
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --mem=320G
#SBATCH --time=3-00:00:00
#SBATCH --output=/home/michaelvolk/Documents/projects/torchcell.worktrees/030-build/experiments/030-solid-growth-multi/slurm/output/%j_%x.out
#SBATCH --error=/home/michaelvolk/Documents/projects/torchcell.worktrees/030-build/experiments/030-solid-growth-multi/slurm/output/%j_%x.out
#SBATCH --no-requeue

# Build the 030 all-allele, multi-measurement solid-growth dataset from the served graph.
#
# MEMORY. 029 peaked at 155 GB of resident memory for 26.8M records (sacct -j 2400),
# and 030 returns about 43.8M, so a linear scaling says about 255 GB. 320 GB leaves
# headroom on a 513 GB node and, at the 4.1 GB MaxMemPerCPU cap, implies an 80-CPU
# floor. What holds memory is aggregation pass 1, one key per record. Nothing else
# large may run beside this; the served Neo4j container is capped at 200 GB but idles
# near 7 GB and grows only through its page cache during the query.
#
# TIME, projected from the measured 029 rates scaled to 43.8M records: raw about 10 h,
# conversion about 1 h 15, aggregation about 6 h 30, processed copy 30 min, label table
# and indices 15 to 20 h. About 33 to 38 h total. The index stage is the least certain:
# it grew fivefold from 025 to 029 because without the merge every record carries all
# its entries, and 030 has both more records and more entries per record than 029.
#
# DISK. Four stage copies. 029 took 615 + 614 + 547 + 547 GB for 26.8M records, so 030
# needs about 3.7 TB. The preflight below refuses to start without it, because a build
# that dies on ENOSPC at hour 20 has wasted the node and left a partial store. Freeing
# it means deleting the 025 and 029 intermediates, both archived and verified under
# /bulk/experiments/*-intermediates/; that deletion is the user's, never this script's.
#
# PRECONDITION, not checked here. DmfKuzmin2018Dataset and DmfKuzmin2020Dataset must be
# re-served with the query-strain fitness records of main 4c4a4f950. Without them 030 is
# 025 without the merge, and the trigenic reconstruction stays capped near 0.54.
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

python experiments/030-solid-growth-multi/scripts/query.py
echo "finished at $(date)"
