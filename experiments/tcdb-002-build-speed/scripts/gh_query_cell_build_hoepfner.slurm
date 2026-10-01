#!/bin/bash
#SBATCH --job-name=tcdb002-cell-hoepfner
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/tcdb-002-build-speed/experiments/tcdb-002-build-speed/slurm/output/%j_%x.out
#SBATCH --error=/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/tcdb-002-build-speed/experiments/tcdb-002-build-speed/slurm/output/%j_%x.out
#SBATCH --no-requeue
# experiments/tcdb-002-build-speed/scripts/gh_query_cell_build_hoepfner.slurm
# [[experiments.tcdb-002-build-speed.scripts.gh_query_cell_build_hoepfner]]
#
# Whole Neo4jCellDataset build of the Hoepfner 2014 block of the 033 query from the
# served graph (query_cell_build_rate.py): partitioned raw stage (8 workers, prefix
# length 2), the 033 aggregator with its pass 1 folded into the raw stage, the processed
# copy and the folded index writer. 3,124,319 served records, 2,880,165 groups in the
# 033 build (its dataset_name_index). The three 40 GB stage stores land on /scratch and
# are removed at the end. 32 GB: the raw store is past the page cache, so the passes
# read from disk as a full build would. Submit after the raw-stage measurement so the
# two do not share the server:
#
#   sbatch --dependency=afterok:<raw-stage job> \
#     experiments/tcdb-002-build-speed/scripts/gh_query_cell_build_hoepfner.slurm
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
ROOT="$DATA_ROOT/data/torchcell/experiments/tcdb-002-build-speed/cell-hoepfner-$SLURM_JOB_ID"
df -h /scratch | tail -1
python experiments/tcdb-002-build-speed/scripts/query_cell_build_rate.py \
  --dataset EnvChemgenHoepfner2014Dataset --gene-set "$GENE_SET" --root "$ROOT" \
  --fetch-workers 8 --prefix-length 2 --expected-groups 2880165 --cleanup || exit 1
df -h /scratch | tail -1
echo "finished at $(date)"
