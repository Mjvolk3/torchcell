#!/bin/bash
# experiments/043-expression-proteome-morphology/scripts/gh_baselines_043.slurm
# [[experiments.043-expression-proteome-morphology.scripts.gh_baselines_043]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/043-expression-proteome-morphology/scripts/gh_baselines_043.slurm
#
# baselines_043.py --stage fit (CPU only, no GPU): constant / additive, ridge, kNN and
# per-gene gradient-boosted trees on the trainer's validation records of every dataset,
# written to results/baselines_043.json. ONLY (optional) restricts to some datasets, e.g.
# ONLY="zelezniak turnover". Submit from the repository root:
#
#   sbatch experiments/043-expression-proteome-morphology/scripts/gh_baselines_043.slurm
#   sbatch --export=ALL,ONLY="hu doubles" experiments/043-expression-proteome-morphology/scripts/gh_baselines_043.slurm
#
# The W&B read (--stage trained) and the figure (--stage report) are light and run after.
#SBATCH -p main
#SBATCH --mem=160g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --job-name=043-baselines
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/projects/torchcell-scratch/experiments/043-expression-proteome-morphology/slurm/output/%x_%j.out
#SBATCH --error=/scratch/projects/torchcell-scratch/experiments/043-expression-proteome-morphology/slurm/output/%x_%j.out
set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate torchcell
cd "${SLURM_SUBMIT_DIR:?submit from the repository root}"
export PYTHONPATH="$PWD"
export EXPERIMENT_ROOT="$PWD/experiments"
export STORE_043_VERSION=v2
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
echo "host $(hostname)  commit $(git rev-parse --short HEAD)  only ${ONLY:-all}  $(date)"
EXTRA=()
if [[ -n "${ONLY:-}" ]]; then read -r -a DS <<< "$ONLY"; EXTRA=(--only "${DS[@]}"); fi
python experiments/043-expression-proteome-morphology/scripts/baselines_043.py --stage fit --n-jobs "${SLURM_CPUS_PER_TASK}" "${EXTRA[@]}"
echo "done $(date)"
