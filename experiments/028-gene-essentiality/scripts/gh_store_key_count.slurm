#!/bin/bash
# experiments/028-gene-essentiality/scripts/gh_store_key_count.slurm
# [[experiments.028-gene-essentiality.scripts.store_key_count]]
#
# Count the distinct (GEM gene subset, medium) keys of the solid-growth build: the
# number of solves the module's store needs. CPU only; a parallel LMDB cursor scan of
# 13.5 million records (517 GB) by regular expression, no JSON parsing.
#
#   ESS_MEM=48g ESS_CPUS=20 bash experiments/028-gene-essentiality/scripts/freeze_and_submit.sh \
#     experiments/028-gene-essentiality/scripts/gh_store_key_count.slurm
#
#SBATCH -p main
#SBATCH --mem=48g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --job-name=028-store-keys
#SBATCH --time=06:00:00
#SBATCH --output=/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/kinetics-equilibrator-datasets/experiments/028-gene-essentiality/slurm/output/%x_%j.out
#SBATCH --error=/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/kinetics-equilibrator-datasets/experiments/028-gene-essentiality/slurm/output/%x_%j.out

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/kinetics-equilibrator-datasets}"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate torchcell
set -a; . "$PROJECT_ROOT/.env"; set +a
export EXPERIMENT_ROOT="$PROJECT_ROOT/experiments"
cd "$PROJECT_ROOT" || exit 1
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS=ignore
export TC_SCAN_WORKERS="${TC_SCAN_WORKERS:-$SLURM_CPUS_PER_TASK}"
echo "host $(hostname); workers $TC_SCAN_WORKERS"
python experiments/028-gene-essentiality/scripts/store_key_count.py
