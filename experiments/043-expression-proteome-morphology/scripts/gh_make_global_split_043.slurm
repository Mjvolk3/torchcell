#!/bin/bash
# experiments/043-expression-proteome-morphology/scripts/gh_make_global_split_043.slurm
# [[experiments.043-expression-proteome-morphology.scripts.gh_make_global_split_043]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/043-expression-proteome-morphology/scripts/gh_make_global_split_043.slurm
#
# Draw the 043 fold file (make_global_split_043.py): reads the split key of every record of
# the 15 datasets by raw index (CPU only, 16 worker processes), then writes
# conf/splits/global_split_v1.json and results/global_split_v1_summary.json. Submit from
# the repository root:
#
#   sbatch experiments/043-expression-proteome-morphology/scripts/gh_make_global_split_043.slurm
#SBATCH -p main
#SBATCH --mem=48g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=043-global-split
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/projects/torchcell-scratch/experiments/043-expression-proteome-morphology/slurm/output/%x_%j.out
#SBATCH --error=/scratch/projects/torchcell-scratch/experiments/043-expression-proteome-morphology/slurm/output/%x_%j.out
# INHERIT_FROM=<split version> (e.g. global_split_v1): inherit its folds, draw only new keys
# (the rule for every split version after v1); unset = full redraw.
set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate torchcell
cd "${SLURM_SUBMIT_DIR:?submit from the repository root}"
export PYTHONPATH="$PWD"
export EXPERIMENT_ROOT="$PWD/experiments"
echo "host $(hostname)  commit $(git rev-parse --short HEAD)  $(date)"
python experiments/043-expression-proteome-morphology/scripts/make_global_split_043.py --workers 16 ${INHERIT_FROM:+--inherit-from "$INHERIT_FROM"}
echo "done $(date)"
