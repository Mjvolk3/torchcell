#!/bin/bash
# experiments/043-expression-proteome-morphology/scripts/gh_within_dataset_ceilings.slurm
# [[experiments.043-expression-proteome-morphology.scripts.gh_within_dataset_ceilings]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/043-expression-proteome-morphology/scripts/gh_within_dataset_ceilings.slurm
#
# Within-dataset reproducibility of the 043 expression datasets
# (within_dataset_ceilings.py): Jackson 2020 construction pairs, Kemmeren 2014
# own-culture test-retest from the raw GEO pickles, the Hu 2007 release check, and the
# Boocock 2025 / N'Guessan 2025 internal checks, then the two figures. CPU only; reads
# dev-tree LMDBs and the Kemmeren raw directory. Submit from the repository root:
#
#   sbatch experiments/043-expression-proteome-morphology/scripts/gh_within_dataset_ceilings.slurm
#   sbatch --export=ALL,STAGE=figures ...   # one stage
#
# Memory: 64 GB. The two Kemmeren GEO pickles are 1.4 GB on disk and several GB as
# GEOparse objects; the Boocock cross-A de novo matrix is 27,744 cells x ~6,000 genes
# in float32 (about 0.7 GB) plus the per-record dicts of the 79,516-record LMDB.
#SBATCH -p main
#SBATCH --mem=64g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=043-within-ceilings
#SBATCH --time=04:00:00
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
python experiments/043-expression-proteome-morphology/scripts/within_dataset_ceilings.py \
    --stage "${STAGE:-all}"
echo "done $(date)"
