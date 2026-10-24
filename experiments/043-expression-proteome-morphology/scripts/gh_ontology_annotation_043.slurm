#!/bin/bash
# experiments/043-expression-proteome-morphology/scripts/gh_ontology_annotation_043.slurm
# [[experiments.043-expression-proteome-morphology.scripts.gh_ontology_annotation_043]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/043-expression-proteome-morphology/scripts/gh_ontology_annotation_043.slurm
#
# CPU: ontology_annotation_043.py over every record of the v2 raw store (15 datasets). Writes the
# vocabulary JSON prep_v4 reads and results/ontology_annotation_043_v2.json.
#
#   sbatch experiments/043-expression-proteome-morphology/scripts/gh_ontology_annotation_043.slurm
#SBATCH -p main
#SBATCH --mem=64g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --job-name=043-onto-annot
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/projects/torchcell-scratch/experiments/043-expression-proteome-morphology/slurm/output/%x_%j.out
#SBATCH --error=/scratch/projects/torchcell-scratch/experiments/043-expression-proteome-morphology/slurm/output/%x_%j.out
set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate torchcell
cd "${SLURM_SUBMIT_DIR:?submit from the repository root}"
export PYTHONPATH="$PWD"
export DATA_ROOT=/scratch/projects/torchcell-scratch
export STORE_043_VERSION=v2
echo "host $(hostname)  commit $(git rev-parse --short HEAD)  $(date)"
python experiments/043-expression-proteome-morphology/scripts/ontology_annotation_043.py --workers 30
echo "done $(date)"
