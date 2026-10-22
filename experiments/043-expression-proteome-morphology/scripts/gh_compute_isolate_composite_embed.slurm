#!/bin/bash
# experiments/043-expression-proteome-morphology/scripts/gh_compute_isolate_composite_embed.slurm
# [[experiments.043-expression-proteome-morphology.scripts.gh_compute_isolate_composite_embed]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/043-expression-proteome-morphology/scripts/gh_compute_isolate_composite_embed.slurm
#
# GPU stage of compute_isolate_composite.py on one card: embed every distinct input not
# yet in each named encoder's allele table. The first argument is the comma list of
# encoders. Each encoder first reproduces a sample of the S288C rows through the batched
# path and stops on disagreement; an input outside the encoder's alphabet is skipped and
# counted before any batch (typed gap ambiguous_sequence, written by --stage alphabet).
# Tables are appended every flush, so a job cut by its limit is resumed by the same line
# (done rows are skipped). Two encoders sets run on two cards in parallel, each with a
# resume job chained afterany; verify runs as a CPU job after both resumes, because it
# reads every table and must not open one another job is still appending to.
#
# Needs the CPU stages first: extract, unpack, align, merge, alphabet, seed.
#
#   sbatch -J 043-iso-embed-A --time=<t> .../gh_compute_isolate_composite_embed.slurm fudt_species_downstream,calm
#   sbatch -J 043-iso-embed-B --time=<t> .../gh_compute_isolate_composite_embed.slurm prot_t5_xl_uniref50
#
# --time per job: from the measured rates (FUDT upstream 76.0 windows per s, job 6106;
# the others in the script note) and the counts left to compute.
#SBATCH -p main
#SBATCH --gres=gpu:1
#SBATCH --mem=48g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --job-name=043-iso-embed
#SBATCH --time=12:00:00
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/scratch/projects/torchcell-scratch/experiments/043-expression-proteome-morphology/slurm/output/%x_%j.out
#SBATCH --error=/scratch/projects/torchcell-scratch/experiments/043-expression-proteome-morphology/slurm/output/%x_%j.out
set -euo pipefail
ENCODERS="${1:?comma list of encoders}"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate torchcell
cd "${SLURM_SUBMIT_DIR:?submit from the repository root}"
export PYTHONPATH="$PWD"
export DATA_ROOT=/scratch/projects/torchcell-scratch
SCRIPT=experiments/043-expression-proteome-morphology/scripts/compute_isolate_composite.py
echo "host $(hostname)  commit $(git rev-parse --short HEAD)  gpu ${CUDA_VISIBLE_DEVICES:-none}  encoders ${ENCODERS}  $(date)"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv
python "$SCRIPT" --stage embed --encoders "${ENCODERS}"
echo "done $(date)"
