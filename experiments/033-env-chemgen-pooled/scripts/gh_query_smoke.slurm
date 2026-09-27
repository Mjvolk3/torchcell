#!/bin/bash
#SBATCH --job-name=033-query-smoke
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/home/michaelvolk/Documents/projects/torchcell.worktrees/exp/033-env-chemgen-pooled/experiments/033-env-chemgen-pooled/slurm/output/%j_%x.out
#SBATCH --error=/home/michaelvolk/Documents/projects/torchcell.worktrees/exp/033-env-chemgen-pooled/experiments/033-env-chemgen-pooled/slurm/output/%j_%x.out
#SBATCH --no-requeue

# Smoke build of the 033 query against the live graph on eight genes, every stage, into a
# scratch root. Anything that touches the served store runs under slurm.
#
# The gene set carries one gene from each dataset's first served record (YAL004W Vanacloig,
# YAL001C Hillenmeyer HET and Hoepfner, YAL013W Wildenhain), the worked-example gene of the
# 031 document that four datasets measure (YBR058C UBP14) and its Wildenhain stand-in
# (YMR263W SAP30), plus Vanacloig's three sensitized-host deletions (YGL013C PDR1,
# YBL005W PDR3, YDR011W SNQ2), without which the ALL(...) gene filter returns no Vanacloig
# record. Each block is a full scan of its dataset even for eight genes.
#
# Submit from the branch worktree; the root is under DATA_ROOT so the summary survives:
#
#   sbatch experiments/033-env-chemgen-pooled/scripts/gh_query_smoke.slurm

echo "Running on $(hostname) at $(date)"
source ~/.zshrc
conda activate ~/miniconda3/envs/torchcell

PROJECT_ROOT="${PROJECT_ROOT:-/home/michaelvolk/Documents/projects/torchcell.worktrees/exp/033-env-chemgen-pooled}"
cd "$PROJECT_ROOT" || exit 1
echo "PROJECT_ROOT=$PROJECT_ROOT ($(git -C "$PROJECT_ROOT" rev-parse --short HEAD 2>/dev/null))"
export PYTHONPATH="$PROJECT_ROOT"

set -a; source "$PROJECT_ROOT/.env"; set +a
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="torchcell"

SMOKE_ROOT="$DATA_ROOT/data/torchcell/experiments/033-env-chemgen-pooled/smoke-8-genes"
if [ -e "$SMOKE_ROOT" ]; then
  echo "ABORT: $SMOKE_ROOT exists; move it aside so the smoke runs every stage." >&2
  exit 1
fi
mkdir -p "$SMOKE_ROOT"

python experiments/033-env-chemgen-pooled/scripts/query.py \
  --root "$SMOKE_ROOT" \
  --genes YAL004W,YBR058C,YAL001C,YAL013W,YMR263W,YGL013C,YBL005W,YDR011W || exit 1
echo "smoke finished at $(date)"
du -sh "$SMOKE_ROOT"/raw "$SMOKE_ROOT"/aggregation "$SMOKE_ROOT"/processed
