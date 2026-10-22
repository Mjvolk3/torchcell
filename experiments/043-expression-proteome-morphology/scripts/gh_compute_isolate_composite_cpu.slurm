#!/bin/bash
# experiments/043-expression-proteome-morphology/scripts/gh_compute_isolate_composite_cpu.slurm
# [[experiments.043-expression-proteome-morphology.scripts.gh_compute_isolate_composite_cpu]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/043-expression-proteome-morphology/scripts/gh_compute_isolate_composite_cpu.slurm
#
# CPU stages of compute_isolate_composite.py (no GPU). The stage is the first argument;
# resources are set per stage on the sbatch line, from measured peaks. Submit from the
# repository root:
#
#   sbatch -J 043-iso-extract --mem=48g --cpus-per-task=2 --time=02:00:00 \
#       experiments/043-expression-proteome-morphology/scripts/gh_compute_isolate_composite_cpu.slurm extract
#   sbatch -J 043-iso-unpack --mem=8g --cpus-per-task=2 --time=01:00:00 \
#       experiments/043-expression-proteome-morphology/scripts/gh_compute_isolate_composite_cpu.slurm unpack
#   sbatch -J 043-iso-align --array=0-63%16 --mem=6g --cpus-per-task=4 --time=04:00:00 \
#       experiments/043-expression-proteome-morphology/scripts/gh_compute_isolate_composite_cpu.slurm align
#   sbatch -J 043-iso-merge --mem=64g --cpus-per-task=2 --time=02:00:00 \
#       experiments/043-expression-proteome-morphology/scripts/gh_compute_isolate_composite_cpu.slurm merge
#   (then seed and verify the same way)
#
# Arguments after the stage pass through to the script, e.g. the CPU check that the
# batched encoder path reproduces the S288C tables (no card, nothing written):
#
#   sbatch -J 043-iso-check --mem=24g --cpus-per-task=16 --time=02:00:00 \
#       experiments/043-expression-proteome-morphology/scripts/gh_compute_isolate_composite_cpu.slurm \
#       check --device cpu --n-check 16
#
# The align array reads SLURM_ARRAY_TASK_ID / SLURM_ARRAY_TASK_COUNT; each task takes
# the isolates cfg.isolates[task::count] and skips an isolate whose part and processing
# record already exist for the same query FASTA sha256, so a resubmit finishes the rest.
#SBATCH -p main
#SBATCH --mem=8g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --job-name=043-iso-cpu
#SBATCH --time=02:00:00
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/scratch/projects/torchcell-scratch/experiments/043-expression-proteome-morphology/slurm/output/%x_%A_%a.out
#SBATCH --error=/scratch/projects/torchcell-scratch/experiments/043-expression-proteome-morphology/slurm/output/%x_%A_%a.out
set -euo pipefail
STAGE="${1:?stage: extract | unpack | align | merge | seed | verify}"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate torchcell
cd "${SLURM_SUBMIT_DIR:?submit from the repository root}"
export PYTHONPATH="$PWD"
export DATA_ROOT=/scratch/projects/torchcell-scratch
# CPU only: hide every card (ProtT5 picks cuda whenever torch can see one)
export CUDA_VISIBLE_DEVICES=""
echo "host $(hostname)  commit $(git rev-parse --short HEAD)  stage ${STAGE}  task ${SLURM_ARRAY_TASK_ID:-none}  $(date)"
/usr/bin/time -v python experiments/043-expression-proteome-morphology/scripts/compute_isolate_composite.py --stage "${STAGE}" "${@:2}"
echo "done $(date)"
