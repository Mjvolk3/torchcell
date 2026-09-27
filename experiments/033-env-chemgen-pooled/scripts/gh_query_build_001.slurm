#!/bin/bash
#SBATCH --job-name=033-query-build
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/michaelvolk/Documents/projects/torchcell.worktrees/033-build/experiments/033-env-chemgen-pooled/slurm/output/%j_%x.out
#SBATCH --error=/home/michaelvolk/Documents/projects/torchcell.worktrees/033-build/experiments/033-env-chemgen-pooled/slurm/output/%j_%x.out
#SBATCH --no-requeue

# Build the pooled four-dataset chemogenomic store from the served graph: raw query ->
# aggregation by (genotype, environment) -> processed -> indices and label table.
#
# SIZE. 6,394,540 served records (kg_manifest.json, release 2026.09.21-ab6d8c5d). One
# served record serializes to 12 to 38 KB with its reference (measured on the dev stores:
# Vanacloig 15 KB, Hillenmeyer HET 12 KB, Hoepfner 13 KB, Wildenhain 38 KB), so the raw
# store is about 90 GB and the three stage copies about 280 GB; the preflight refuses to
# start without 350 GB free on /db.
#
# MEMORY. Aggregation pass 1 holds one key per record, 6.4M sha256 strings plus the input
# keys, about 1.5 GB; the reference-index and label-table passes map the store through
# LMDB and the cgroup charges the page cache to the job, so resident memory climbs to the
# request and stays there without being need. 64 GB at the 4.1 GB MaxMemPerCPU cap
# implies 16 CPUs.
#
# TIME. The raw stage is the single-threaded Neo4jQueryRaw of main, one commit per
# record, measured at about 630 records/s on the 025 build: about 3 h. Aggregation pass 1
# parses each record's environment once at 0.12 to 0.36 ms (measured on the dev stores):
# under 1 h, and the join pass is I/O. Indices and label table: the 029 build's index
# stage ran 9.5 h over 26.8M records, so 2 to 3 h here.
#
# DISK. The dataset root lives on /db (a build is regenerable) behind a symlink under
# $DATA_ROOT, as the 025, 029 and 030 builds do.
#
# The job reads its code from a DETACHED worktree pinned at the build commit, so the
# branch worktree can keep moving while the build runs:
#
#   git worktree add --detach ~/Documents/projects/torchcell.worktrees/033-build <sha>
#   cp <branch worktree>/.env ~/Documents/projects/torchcell.worktrees/033-build/.env
#   mkdir -p ~/Documents/projects/torchcell.worktrees/033-build/experiments/033-env-chemgen-pooled/slurm/output
#   sbatch ~/Documents/projects/torchcell.worktrees/033-build/experiments/033-env-chemgen-pooled/scripts/gh_query_build_001.slurm

echo "Running on $(hostname) at $(date)"
source ~/.zshrc
conda activate ~/miniconda3/envs/torchcell

PROJECT_ROOT="${PROJECT_ROOT:-/home/michaelvolk/Documents/projects/torchcell.worktrees/033-build}"
cd "$PROJECT_ROOT" || exit 1
echo "PROJECT_ROOT=$PROJECT_ROOT ($(git -C "$PROJECT_ROOT" rev-parse --short HEAD 2>/dev/null))"
export PYTHONPATH="$PROJECT_ROOT"

set -a; source "$PROJECT_ROOT/.env"; set +a
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="torchcell"

NEED_GB="${NEED_GB:-350}"
FREE_GB=$(df -BG --output=avail /db | tail -1 | tr -dc '0-9')
echo "/db free ${FREE_GB} GB; the three stage copies need about ${NEED_GB} GB"
if [ "$FREE_GB" -lt "$NEED_GB" ]; then
  echo "ABORT: not enough space on /db." >&2
  exit 1
fi

DB_ROOT=/db/experiments/033-env-chemgen-pooled-001-pooled-build
LINK="$DATA_ROOT/data/torchcell/experiments/033-env-chemgen-pooled/001-pooled-build"
mkdir -p "$DB_ROOT" "$(dirname "$LINK")"
[[ -e "$LINK" ]] || ln -s "$DB_ROOT" "$LINK"
echo "dataset root: $LINK -> $(readlink -f "$LINK")"
df -h /db | tail -1

python experiments/033-env-chemgen-pooled/scripts/query.py || exit 1
echo "build finished at $(date)"
du -sh "$DB_ROOT"/raw "$DB_ROOT"/aggregation "$DB_ROOT"/processed 2>/dev/null
df -h /db | tail -1
