#!/bin/bash
# experiments/043-expression-proteome-morphology/scripts/gh_reindex_043_store.slurm
# [[experiments.043-expression-proteome-morphology.scripts.gh_reindex_043_store]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/043-expression-proteome-morphology/scripts/gh_reindex_043_store.slurm
#
# Recompute the experiment 043 store's index files from the existing processed LMDB and
# read one sample per dataset (reindex_043_store.py). No query, no rebuild. Move the stale
# index json aside first. Under slurm, not the login shell. Submit from the repository root:
#
#   sbatch experiments/043-expression-proteome-morphology/scripts/gh_reindex_043_store.slurm
#
# Memory: 64 GB; the index passes stream the processed LMDB one entry at a time.
#SBATCH -p main
#SBATCH --mem=64g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=043-reindex-store
#SBATCH --time=06:00:00
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/scratch/projects/torchcell-scratch/experiments/043-expression-proteome-morphology/slurm/output/%x_%j.out
#SBATCH --error=/scratch/projects/torchcell-scratch/experiments/043-expression-proteome-morphology/slurm/output/%x_%j.out
set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate torchcell
cd "${SLURM_SUBMIT_DIR:?submit from the repository root}"
export PYTHONPATH="$PWD"
echo "host $(hostname)  commit $(git rev-parse --short HEAD)  $(date)"
python experiments/043-expression-proteome-morphology/scripts/reindex_043_store.py
echo "done $(date)"
