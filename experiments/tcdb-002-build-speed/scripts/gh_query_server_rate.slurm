#!/bin/bash
#SBATCH --job-name=tcdb002-query-rate
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/tcdb-002-build-speed/experiments/tcdb-002-build-speed/slurm/output/%j_%x.out
#SBATCH --error=/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/tcdb-002-build-speed/experiments/tcdb-002-build-speed/slurm/output/%j_%x.out
#SBATCH --no-requeue
# experiments/tcdb-002-build-speed/scripts/gh_query_server_rate.slurm
# [[experiments.tcdb-002-build-speed.scripts.gh_query_server_rate]]
#
# Server and driver side of a query build, measured on the Vanacloig 2022 block of the
# 033 query against the live graph (query_server_rate.py). Anything that touches the
# served store runs under slurm. The gene set is the 033 build's own gene_set.json.
#
#   sbatch experiments/tcdb-002-build-speed/scripts/gh_query_server_rate.slurm
echo "Running on $(hostname) at $(date)"
source ~/.zshrc
conda activate ~/miniconda3/envs/torchcell
PROJECT_ROOT="/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/tcdb-002-build-speed"
cd "$PROJECT_ROOT" || exit 1
echo "PROJECT_ROOT=$PROJECT_ROOT ($(git -C "$PROJECT_ROOT" rev-parse --short HEAD 2>/dev/null))"
export PYTHONPATH="$PROJECT_ROOT"
set -a; source "$PROJECT_ROOT/.env"; set +a
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="torchcell"
export WANDB_MODE=disabled
GENE_SET="$DATA_ROOT/data/torchcell/experiments/033-env-chemgen-pooled/001-pooled-build/processed/gene_set.json"
SCRATCH_ROOT="$DATA_ROOT/data/torchcell/experiments/tcdb-002-build-speed/query-rate-$SLURM_JOB_ID"
mkdir -p "$SCRATCH_ROOT"
python experiments/tcdb-002-build-speed/scripts/query_server_rate.py \
  --gene-set "$GENE_SET" --scratch-root "$SCRATCH_ROOT" || exit 1
echo "finished at $(date)"
