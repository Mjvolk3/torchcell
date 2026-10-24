#!/bin/bash
# experiments/043-expression-proteome-morphology/scripts/gh_train_043_prepare.slurm
# [[experiments.043-expression-proteome-morphology.scripts.gh_train_043_prepare]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/043-expression-proteome-morphology/scripts/gh_train_043_prepare.slurm
#
# CPU: train_043.py --stage prepare. Reads every record of the trainer's datasets from the raw
# 043 store with the route-A representation (keys only, no vectors), drops the test fold,
# writes the prepared store under
# $DATA_ROOT/data/torchcell/experiments/043-expression-proteome-morphology/train_043_prep/prep_<PREP_VERSION>_routeA/ (prep_v2 since 2026-10-23: adds Hu, Jackson, Nadal-Ribelles)
# The store-wide verification (job 6120) read 173,617 records in 3 min 37 s on 32 workers; the
# graphs and the Boocock raw-byte filter add to that, so 2 h is generous.
#
#   sbatch experiments/043-expression-proteome-morphology/scripts/gh_train_043_prepare.slurm
#
# prep_v4 (2026-10-23: prep_v3 plus the experiment ontology annotation; the vocabulary comes
# from gh_ontology_annotation_043.slurm):
#   sbatch --export=ALL,STORE_043_VERSION=v2,PREP_VERSION=prep_v4,ONTOLOGY_VOCAB=$DATA_ROOT/data/torchcell/experiments/043-expression-proteome-morphology/ontology_043_v2/experiment_ontology_vocab.json \
#     experiments/043-expression-proteome-morphology/scripts/gh_train_043_prepare.slurm
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
PREP_ARGS=()
if [ -n "${PREP_VERSION:-}" ]; then PREP_ARGS+=(--prep-version "$PREP_VERSION"); fi
if [ -n "${ONTOLOGY_VOCAB:-}" ]; then PREP_ARGS+=(--ontology-vocab "$ONTOLOGY_VOCAB"); fi
echo "store ${STORE_043_VERSION:-unset}  prep ${PREP_VERSION:-generation default}  vocab ${ONTOLOGY_VOCAB:-none}"
python experiments/043-expression-proteome-morphology/scripts/train_043.py --stage prepare --route A --workers 30 "${PREP_ARGS[@]}"
echo "done $(date)"
