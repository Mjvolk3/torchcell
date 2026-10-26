#!/bin/bash
# experiments/043-expression-proteome-morphology/scripts/gh_diagnostics_043.slurm
# [[experiments.043-expression-proteome-morphology.scripts.gh_diagnostics_043]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/043-expression-proteome-morphology/scripts/gh_diagnostics_043.slurm
#
# diagnostics_043.py on CPU (no GPU): D1 (doubles composition, forward passes of 5 runs'
# checkpoints over 72 doubles + 144 singles) and D2 (isolate read-outs of the 3 bs16
# checkpoints over Caudal and Muenzner, one trunk pass per isolate context, then ridge).
# Store generation v2. Submit from the repository root; STAGES picks the stages (default
# "d1 d2"); the wandb and report stages run on the login shell (light):
#
#   sbatch experiments/043-expression-proteome-morphology/scripts/gh_diagnostics_043.slurm
#   sbatch --export=ALL,STAGES=d2,ARGS="--smoke",SMOKE_JSON=/tmp/x.json ...
#SBATCH -p main
#SBATCH --mem=200g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --job-name=043-diagnostics
#SBATCH --time=08:00:00
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
echo "host $(hostname)  commit $(git rev-parse --short HEAD)  stages ${STAGES:-d1 d2}  args ${ARGS:-}  $(date)"
for stage in ${STAGES:-d1 d2}; do
  # shellcheck disable=SC2086
  python experiments/043-expression-proteome-morphology/scripts/diagnostics_043.py --stage "$stage" ${ARGS:-}
done
echo "done $(date)"
