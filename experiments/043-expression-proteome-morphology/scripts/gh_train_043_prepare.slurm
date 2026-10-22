#!/bin/bash
# experiments/043-expression-proteome-morphology/scripts/gh_train_043_prepare.slurm
# [[experiments.043-expression-proteome-morphology.scripts.gh_train_043_prepare]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/043-expression-proteome-morphology/scripts/gh_train_043_prepare.slurm
#
# CPU: train_043.py --stage prepare. Reads every record of the trainer's datasets from the raw
# 043 store with the route-A representation (keys only, no vectors), drops the test fold,
# writes the prepared store under
# $DATA_ROOT/data/torchcell/experiments/043-expression-proteome-morphology/train_043_prep/prep_v1_routeA/
# The store-wide verification (job 6120) read 173,617 records in 3 min 37 s on 32 workers; the
# graphs and the Boocock raw-byte filter add to that, so 2 h is generous.
#
#   sbatch experiments/043-expression-proteome-morphology/scripts/gh_train_043_prepare.slurm
#SBATCH -p main
#SBATCH --mem=96g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --job-name=043-train-prep
#SBATCH --time=02:00:00
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
export EXPERIMENT_ROOT="$PWD/experiments"
echo "host $(hostname)  commit $(git rev-parse --short HEAD)  $(date)"
python experiments/043-expression-proteome-morphology/scripts/train_043.py --stage prepare --route A --workers 30
echo "done $(date)"
