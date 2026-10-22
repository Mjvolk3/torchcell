#!/bin/bash
# experiments/043-expression-proteome-morphology/scripts/gh_resolve_segregant_alleles_043.slurm
# [[experiments.043-expression-proteome-morphology.scripts.gh_resolve_segregant_alleles_043]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/043-expression-proteome-morphology/scripts/gh_resolve_segregant_alleles_043.slurm
#
# CPU: resolve_segregant_alleles_043.py over all Boocock 2025 and N'Guessan 2025 records
# of the raw store (82,467), 32 forked workers. Needs representation_043.py --stage build.
# A Boocock record parses in about 46 ms (measured in the session on 50 records), so the
# pass is a few minutes; 1 h covers a cold page cache on the 163 GB LMDB.
#
#   sbatch experiments/043-expression-proteome-morphology/scripts/gh_resolve_segregant_alleles_043.slurm
#SBATCH -p main
#SBATCH --mem=64g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --job-name=043-seg-alleles
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
python experiments/043-expression-proteome-morphology/scripts/resolve_segregant_alleles_043.py --workers 32
echo "done $(date)"
