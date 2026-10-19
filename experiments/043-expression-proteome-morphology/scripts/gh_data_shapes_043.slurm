#!/bin/bash
# experiments/043-expression-proteome-morphology/scripts/gh_data_shapes_043.slurm
# [[experiments.043-expression-proteome-morphology.scripts.gh_data_shapes_043]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/043-expression-proteome-morphology/scripts/gh_data_shapes_043.slurm
#
# Full pass of data_shapes_043.py over the 043 raw store (every one of the 173,617
# records; CPU only, the store is read, not queried). The head-only raw dataset index
# is reused when raw/raw_dataset_name_index.json already exists. Submit from the
# repository root:
#
#   sbatch experiments/043-expression-proteome-morphology/scripts/gh_data_shapes_043.slurm
#
# Memory: 16 GB. One record is decoded at a time; the tallies hold per-record
# scalars plus the union of label keys per dataset.
#SBATCH -p main
#SBATCH --mem=16g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --job-name=043-data-shapes
#SBATCH --time=03:00:00
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
python experiments/043-expression-proteome-morphology/scripts/data_shapes_043.py --mode full
echo "done $(date)"
