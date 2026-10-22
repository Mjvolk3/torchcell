#!/bin/bash
# experiments/043-expression-proteome-morphology/scripts/gh_verify_representation_043.slurm
# [[experiments.043-expression-proteome-morphology.scripts.gh_verify_representation_043]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/043-expression-proteome-morphology/scripts/gh_verify_representation_043.slurm
#
# CPU: verify_representation_043.py over all 173,617 records of the raw store, 32 forked
# workers, keys only. Chain it after the segregant allele pass so the cross-check reads
# segregant_alleles.parquet:
#
#   sbatch --dependency=afterok:<resolve_segregant_alleles job> experiments/043-expression-proteome-morphology/scripts/gh_verify_representation_043.slurm
#
# The segregant pass (82,467 records, same per-record cost) ran in about 3 min on 32
# workers; 1 h covers the rest of the store and a cold page cache.
#SBATCH -p main
#SBATCH --mem=64g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --job-name=043-rep-verify
#SBATCH --time=01:00:00
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/scratch/projects/torchcell-scratch/experiments/043-expression-proteome-morphology/slurm/output/%x_%j.out
#SBATCH --error=/scratch/projects/torchcell-scratch/experiments/043-expression-proteome-morphology/slurm/output/%x_%j.out
set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate torchcell
cd "${SLURM_SUBMIT_DIR:?submit from the repository root}"
export PYTHONPATH="$PWD"
export DATA_ROOT=/scratch/projects/torchcell-scratch
echo "host $(hostname)  commit $(git rev-parse --short HEAD)  $(date)"
python experiments/043-expression-proteome-morphology/scripts/verify_representation_043.py --workers 32
echo "done $(date)"
