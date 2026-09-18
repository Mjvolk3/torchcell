#!/bin/bash
#SBATCH --job-name=029-query-build
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=160G
#SBATCH --time=3-00:00:00
#SBATCH --output=/home/michaelvolk/Documents/projects/torchcell.worktrees/029-build/experiments/029-solid-growth-ko/slurm/output/%j_%x.out
#SBATCH --error=/home/michaelvolk/Documents/projects/torchcell.worktrees/029-build/experiments/029-solid-growth-ko/slurm/output/%j_%x.out
#SBATCH --no-requeue

# Build the 029 deletion-only solid-growth dataset from the served graph.
#
# Memory: the 025 build at 100 G died three times (job 1559 at the end of the raw
# fetch, 1560 after conversion, 1561 in a later stage) and finished on the fourth
# submission, so this asks for 160 G. The stages that held memory in 025 -- the
# per-record write transactions, the reference-index pass, the dedup maps -- are
# gone or batched (commit 2824b6335), and there is no dedup stage here at all;
# what remains resident is aggregation pass 1 (one key per record, about 28M) and
# the label table. MaxMemPerCPU=4100 makes 160 G imply a 40-CPU floor.
#
# Stage timings of 025 for reference (43.8M records, four jobs): raw 17 h 55 min,
# conversion 13 h 36 min, dedup 10 h 52 min, aggregation 3 h 43 min, processed 16
# min, indices and label table 2 h 50 min. Projection for 029 (about 28M records,
# passthrough conversion, no dedup, byte-keyed aggregation): raw 8 to 12 h,
# conversion under 1 h, aggregation 1 to 2 h, indices 2 to 3 h.
#
# The dataset root symlinks to /db (the SN850X): a build is regenerable, which is
# the /db rule, and /scratch has no room for the stage copies.
#
# The job reads its code from a DETACHED worktree pinned at the build commit
# (~/Documents/projects/torchcell.worktrees/029-build), so the branch worktree can keep
# moving while the build runs:
#
#   git worktree add --detach ~/Documents/projects/torchcell.worktrees/029-build <sha>
#   cp <branch worktree>/.env ~/Documents/projects/torchcell.worktrees/029-build/.env
#   sbatch ~/Documents/projects/torchcell.worktrees/029-build/experiments/029-solid-growth-ko/scripts/gh_query_build_001.slurm

echo "Running on $(hostname) at $(date)"
source ~/.zshrc
conda activate ~/miniconda3/envs/torchcell

PROJECT_ROOT="${PROJECT_ROOT:-/home/michaelvolk/Documents/projects/torchcell.worktrees/029-build}"
cd "$PROJECT_ROOT" || exit 1
echo "PROJECT_ROOT=$PROJECT_ROOT ($(git -C "$PROJECT_ROOT" rev-parse --short HEAD 2>/dev/null))"
export PYTHONPATH="$PROJECT_ROOT"

set -a; source "$PROJECT_ROOT/.env"; set +a
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="torchcell"

DB_ROOT=/db/experiments/029-solid-growth-ko-001-ko-build
LINK="$DATA_ROOT/data/torchcell/experiments/029-solid-growth-ko/001-ko-build"
mkdir -p "$DB_ROOT" "$(dirname "$LINK")"
[[ -e "$LINK" ]] || ln -s "$DB_ROOT" "$LINK"
echo "dataset root: $LINK -> $(readlink -f "$LINK")"
df -h /db | tail -1

python experiments/029-solid-growth-ko/scripts/query.py
echo "finished at $(date)"
