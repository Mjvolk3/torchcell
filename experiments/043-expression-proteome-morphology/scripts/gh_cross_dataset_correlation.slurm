#!/bin/bash
# experiments/043-expression-proteome-morphology/scripts/gh_cross_dataset_correlation.slurm
# [[experiments.043-expression-proteome-morphology.scripts.gh_cross_dataset_correlation]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/043-expression-proteome-morphology/scripts/gh_cross_dataset_correlation.slurm
#
# Cross-dataset correlations of the 043 raw store (cross_dataset_correlation.py): one read
# of the 163 GB raw LMDB (CPU only; the store is read, not queried), then the pair
# statistics, the morphology ridge and the figures. Submit from the repository root:
#
#   sbatch experiments/043-expression-proteome-morphology/scripts/gh_cross_dataset_correlation.slurm
#
# Memory: 48 GB. The extraction holds the label values of about 25,000 records of the 11
# compared datasets as Python dicts (about 70 M values, a few GB) before they become
# float32 frames. The --moving-features list (116 CalMorph features) is read from the
# multimodal-phenotype-retrospective worktree, recorded with its sha256 in the output.
#SBATCH -p main
#SBATCH --mem=48g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=043-cross-dataset-corr
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
MOVING=/home/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/results/calmorph_moving_features.json
echo "host $(hostname)  commit $(git rev-parse --short HEAD)  $(date)"
python experiments/043-expression-proteome-morphology/scripts/cross_dataset_correlation.py \
    --stage all --moving-features "$MOVING"
echo "done $(date)"
