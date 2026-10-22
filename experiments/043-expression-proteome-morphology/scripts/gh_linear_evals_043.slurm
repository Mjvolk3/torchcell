#!/bin/bash
# experiments/043-expression-proteome-morphology/scripts/gh_linear_evals_043.slurm
# [[experiments.043-expression-proteome-morphology.scripts.gh_linear_evals_043]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/043-expression-proteome-morphology/scripts/gh_linear_evals_043.slurm
#
# One stage of linear_evals_043.py (CPU only). STAGE is extract | deletion | doubles |
# isolates | segregants | vaishnav | morphology | report; DATASET names the split-A dataset
# for STAGE=deletion. Submit from the repository root, one sbatch per stage:
#
#   sbatch --export=ALL,STAGE=extract experiments/043-expression-proteome-morphology/scripts/gh_linear_evals_043.slurm
#   sbatch --export=ALL,STAGE=deletion,DATASET=kemmeren --job-name=043-lin-kemmeren ...
#
# extract reads the store with 16 worker processes (the split draw, job 6046, peaked at a
# summed 72 GB RSS over 16 forked readers on 48 GB of cgroup memory); the ladder stages are
# BLAS-bound and use the CPUs of the job through OMP_NUM_THREADS.
#SBATCH -p main
#SBATCH --mem=48g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=043-linear-evals
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/projects/torchcell-scratch/experiments/043-expression-proteome-morphology/slurm/output/%x_%j.out
#SBATCH --error=/scratch/projects/torchcell-scratch/experiments/043-expression-proteome-morphology/slurm/output/%x_%j.out
set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate torchcell
cd "${SLURM_SUBMIT_DIR:?submit from the repository root}"
export PYTHONPATH="$PWD"
export EXPERIMENT_ROOT="$PWD/experiments"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
echo "host $(hostname)  commit $(git rev-parse --short HEAD)  stage ${STAGE:?} ${DATASET:-}  $(date)"
EXTRA=()
if [[ -n "${DATASET:-}" ]]; then EXTRA=(--dataset "$DATASET"); fi
python experiments/043-expression-proteome-morphology/scripts/linear_evals_043.py --stage "$STAGE" "${EXTRA[@]}"
echo "done $(date)"
