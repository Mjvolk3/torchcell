#!/bin/bash
#SBATCH --job-name=build_dataset_lmdb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/projects/torchcell/database/slurm/output/%j_%x.out
#SBATCH --error=/scratch/projects/torchcell/database/slurm/output/%j_%x.out
#SBATCH --no-requeue

# Build ONE dataset's LMDB in the DEV tree ($DATA_ROOT/data/torchcell/<slug>) so it can be
# staged into the knowledge-graph build tree and admitted incrementally.
#
#   sbatch --export=ALL,DATASET_CLASS=NadalRibellesPerturbSeq2025Dataset \
#       database/slurm/scripts/gilahyper_build_dataset_lmdb.slurm
#
# TORCHCELL_SRC (optional) points at the checkout whose code should build the dataset,
# e.g. a worktree; it is put on PYTHONPATH ahead of the installed package. DATA_ROOT comes
# from that checkout's .env (the dev tree, torchcell-scratch), never the build tree.
# A pre-existing processed/lmdb makes the build REFUSE (the loader would silently reuse
# it); retire it with scripts/deprecate.sh first.
# Sizing: Nadal-Ribelles 2025 (6188 pseudobulk records, 426 MB R object) peaked at
# 45.6 G RSS and took 10.6 min (job 1688), so 64 G is the floor for that family.

set -euo pipefail
echo "Running on $(hostname) at $(date)"
# conda.sh, not ~/.zshrc: the zsh rc has interactive-only lines (bindkey) that fail
# under bash and would trip `set -e`.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate torchcell

: "${DATASET_CLASS:?set DATASET_CLASS=<registered dataset class name>}"
TORCHCELL_SRC="${TORCHCELL_SRC:-/home/michaelvolk/Documents/projects/torchcell}"
cd "$TORCHCELL_SRC"
echo "source: $TORCHCELL_SRC ($(git rev-parse --short HEAD 2>/dev/null || echo not-a-git-repo))"
echo "dataset: $DATASET_CLASS"

PYTHONPATH="$TORCHCELL_SRC" python -m torchcell.database.build_dataset_lmdb \
    --dataset "$DATASET_CLASS" --io-workers "$SLURM_CPUS_PER_TASK"
echo "DONE at $(date)"
