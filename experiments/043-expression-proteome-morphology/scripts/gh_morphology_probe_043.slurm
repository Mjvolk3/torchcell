#!/bin/bash
# experiments/043-expression-proteome-morphology/scripts/gh_morphology_probe_043.slurm
# [[experiments.043-expression-proteome-morphology.scripts.gh_morphology_probe_043]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/043-expression-proteome-morphology/scripts/gh_morphology_probe_043.slurm
#
# morphology_probe_043.py on CPU (no GPU: the forward passes of the two checkpoints over
# the Ohya 2005 strains and the ridge fits). Store generation v2. Submit from the
# repository root; ARGS passes extra flags (e.g. ARGS=--only-frozen):
#
#   sbatch experiments/043-expression-proteome-morphology/scripts/gh_morphology_probe_043.slurm
#SBATCH -p main
#SBATCH --mem=96g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --job-name=043-morph-probe
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/projects/torchcell-scratch/experiments/043-expression-proteome-morphology/slurm/output/%x_%j.out
#SBATCH --error=/scratch/projects/torchcell-scratch/experiments/043-expression-proteome-morphology/slurm/output/%x_%j.out
set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate torchcell
cd "${SLURM_SUBMIT_DIR:?submit from the repository root}"
export PYTHONPATH="$PWD"
export EXPERIMENT_ROOT="$PWD/experiments"
export ASSET_IMAGES_DIR="$PWD/notes/assets/images"
export STORE_043_VERSION=v2
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
echo "host $(hostname)  commit $(git rev-parse --short HEAD)  args ${ARGS:-}  $(date)"
# shellcheck disable=SC2086
python experiments/043-expression-proteome-morphology/scripts/morphology_probe_043.py ${ARGS:-}
echo "done $(date)"
