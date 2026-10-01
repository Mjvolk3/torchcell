#!/bin/bash
#SBATCH --job-name=tcdb002-fetch-hoepfner
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/tcdb-002-build-speed/experiments/tcdb-002-build-speed/slurm/output/%j_%x.out
#SBATCH --error=/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/tcdb-002-build-speed/experiments/tcdb-002-build-speed/slurm/output/%j_%x.out
#SBATCH --no-requeue
# experiments/tcdb-002-build-speed/scripts/gh_query_fetch_workers_hoepfner.slurm
# [[experiments.tcdb-002-build-speed.scripts.gh_query_fetch_workers_hoepfner]]
#
# Partitioned raw stage on the Hoepfner 2014 block of the 033 query (3,124,319 served
# records, half the 033 build) against the live graph: one-session baseline, then
# fetch_workers 8 at partition_prefix_length 2 (256 partitions per block), the setting a
# multi-million-record block needs, checked byte-identical to the baseline by a streamed
# digest (query_server_rate.py --fetch-workers). Each arm's 40 GB raw store is removed
# once hashed. Baseline at the measured 1,806 records/s is about 29 min; the time limit
# covers both arms. 10 CPUs: 8 workers + the writing parent + slack; 32 GB bounds the
# in-flight partitions (16 of about 12k rendered records each).
#
#   sbatch experiments/tcdb-002-build-speed/scripts/gh_query_fetch_workers_hoepfner.slurm
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
SCRATCH_ROOT="$DATA_ROOT/data/torchcell/experiments/tcdb-002-build-speed/fetch-hoepfner-$SLURM_JOB_ID"
mkdir -p "$SCRATCH_ROOT"
df -h /scratch | tail -1
python experiments/tcdb-002-build-speed/scripts/query_server_rate.py \
  --gene-set "$GENE_SET" --scratch-root "$SCRATCH_ROOT" \
  --dataset EnvChemgenHoepfner2014Dataset --fetch-workers 8 --prefix-length 2 --cleanup \
  --out experiments/tcdb-002-build-speed/results/query_server_rate_fetch_workers_hoepfner.csv || exit 1
df -h /scratch | tail -1
echo "finished at $(date)"
