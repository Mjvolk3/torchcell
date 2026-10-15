#!/bin/bash
#SBATCH --job-name=prebuild-verify-served-dump
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/projects/torchcell-scratch/dev/prebuild-verify/slurm-output/%j_%x.out
#SBATCH --error=/scratch/projects/torchcell-scratch/dev/prebuild-verify/slurm-output/%j_%x.out
#SBATCH --no-requeue
# experiments/database/scripts/prebuild_verify_served_dump.slurm
# [[experiments.database.scripts.prebuild_verify_served_dump]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/database/scripts/prebuild_verify_served_dump.slurm
#
# Read-only dump of the served KG 3.0 records of the lane P sample through bolt
# (prebuild_verify_served_dump.py). Under slurm because every job that touches the
# served store is scheduled, never a shell process.
#
#   sbatch --export=ALL,TORCHCELL_SRC=<worktree>,OUT_DIR=<dir>,DATASETS=<a,b,c> \
#       experiments/database/scripts/prebuild_verify_served_dump.slurm
set -euo pipefail
echo "Running on $(hostname) at $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate torchcell
: "${TORCHCELL_SRC:?set TORCHCELL_SRC=<checkout>}"
: "${OUT_DIR:?set OUT_DIR=<directory for the JSONL files>}"
: "${DATASETS:?set DATASETS=<comma-separated dataset classes>}"
cd "$TORCHCELL_SRC"
/usr/bin/time -v env PYTHONPATH="$TORCHCELL_SRC" python \
    experiments/database/scripts/prebuild_verify_served_dump.py \
    --datasets "$DATASETS" --out-dir "$OUT_DIR"
echo "DONE at $(date)"
