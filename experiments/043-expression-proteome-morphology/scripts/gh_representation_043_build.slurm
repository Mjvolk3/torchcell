#!/bin/bash
# experiments/043-expression-proteome-morphology/scripts/gh_representation_043_build.slurm
# [[experiments.043-expression-proteome-morphology.scripts.gh_representation_043_build]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/043-expression-proteome-morphology/scripts/gh_representation_043_build.slurm
#
# CPU: representation_043.py --stage build. Reads the isolate composite index (6.17 M
# rows, six row groups) and writes the key vocabularies and per-context key tables
# under $DATA_ROOT/data/torchcell/experiments/043-expression-proteome-morphology/representation_043/.
# Keys only; needs no embedding vector, so it runs while the GPU embed jobs are queued.
#
#   sbatch experiments/043-expression-proteome-morphology/scripts/gh_representation_043_build.slurm
#SBATCH -p main
#SBATCH --mem=48g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=043-rep-build
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
python experiments/043-expression-proteome-morphology/scripts/representation_043.py --stage build
echo "done $(date)"
