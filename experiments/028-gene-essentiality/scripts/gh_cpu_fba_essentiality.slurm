#!/bin/bash
# experiments/028-gene-essentiality/scripts/gh_cpu_fba_essentiality.slurm
# [[experiments.028-gene-essentiality.scripts.fba_essentiality_baseline]]
#
# FBA single-gene-deletion essentiality on yeast-GEM 9.0.2 under four media. CPU only;
# 1,121 deletions x 4 media at 4 processes is minutes.
#
#   sbatch experiments/028-gene-essentiality/scripts/gh_cpu_fba_essentiality.slurm
#
#SBATCH -p main
#SBATCH --mem=8g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=028-fba-ess
#SBATCH --time=02:00:00
#SBATCH --output=/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/028-gene-essentiality/experiments/028-gene-essentiality/slurm/output/%x_%j.out
#SBATCH --error=/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/028-gene-essentiality/experiments/028-gene-essentiality/slurm/output/%x_%j.out

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/028-gene-essentiality}"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate torchcell
set -a; . "$PROJECT_ROOT/.env"; set +a
export EXPERIMENT_ROOT="$PROJECT_ROOT/experiments"
cd "$PROJECT_ROOT" || exit 1
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS=ignore
python experiments/028-gene-essentiality/scripts/fba_essentiality_baseline.py
