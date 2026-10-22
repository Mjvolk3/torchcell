#!/bin/bash
# experiments/043-expression-proteome-morphology/scripts/gh_embed_vaishnav_reporter_043.slurm
# [[experiments.043-expression-proteome-morphology.scripts.gh_embed_vaishnav_reporter_043]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/043-expression-proteome-morphology/scripts/gh_embed_vaishnav_reporter_043.slurm
#
# GPU stage of embed_vaishnav_reporter_043.py: the 71,158 Vaishnav upstream windows
# through FUDT species_upstream, the one constant downstream window through FUDT
# species_downstream, and the reporter (yEVenus) protein and CDS through ProtT5 and
# CaLM, appended to the isolate allele tables under their sha256; then the verify stage.
# The CPU `inputs` stage must have run (it writes sequences/vaishnav2022_inputs.parquet).
#
# Chained after the isolate embed job so only one 043 GPU job holds a card and the two
# never append to the same h5 file at once:
#
#   sbatch --dependency=afterany:6106 experiments/043-expression-proteome-morphology/scripts/gh_embed_vaishnav_reporter_043.slurm
#
# --time 1.5 h. Job 6106 measured FUDT upstream at 73.9 windows per s on one RTX 6000
# (76,800 windows in 1,040 s), so 71,158 windows take about 16 min; the four reference
# checks and model loads add about 10 min. A job cut by the limit is resumed by
# resubmitting (done rows skip).
#SBATCH -p main
#SBATCH --gres=gpu:1
#SBATCH --mem=32g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=043-vaish-embed
#SBATCH --time=01:30:00
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
SCRIPT=experiments/043-expression-proteome-morphology/scripts/embed_vaishnav_reporter_043.py
echo "host $(hostname)  commit $(git rev-parse --short HEAD)  gpu ${CUDA_VISIBLE_DEVICES:-none}  $(date)"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv
python "$SCRIPT" --stage embed
python "$SCRIPT" --stage verify
echo "done $(date)"
