#!/bin/bash
# experiments/043-expression-proteome-morphology/scripts/gh_compute_isolate_composite_embed.slurm
# [[experiments.043-expression-proteome-morphology.scripts.gh_compute_isolate_composite_embed]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/043-expression-proteome-morphology/scripts/gh_compute_isolate_composite_embed.slurm
#
# GPU stage of compute_isolate_composite.py: embed every distinct input not yet in its
# allele table, one card, the three cheap encoders first (FUDT upstream, CaLM, FUDT
# downstream) and ProtT5-XL last, then the verify stage. Each encoder first reproduces
# a sample of the S288C rows through the batched path and stops on disagreement. The
# tables are appended every flush, so a job cut by the time limit is resumed by
# resubmitting this script (done rows are skipped).
#
# Needs the CPU stages first: extract, unpack, align, merge, seed. Submitted HELD (the
# cards are held by the 031 packs); release with `scontrol release <jobid>`:
#
#   sbatch --hold experiments/043-expression-proteome-morphology/scripts/gh_compute_isolate_composite_embed.slurm
#
# --time 18 h. Distinct inputs still to compute after the S288C seed (merge job 6089,
# seed job 6088, measured): FUDT upstream ~1.131 M, CaLM ~1.157 M, FUDT downstream
# ~0.407 M, ProtT5 ~0.751 M. Per-card rates on one RTX 6000 Ada are ASSUMED, not
# measured (hypothesis): FUDT upstream fp32 150-300 per s, CaLM fp32 300-600 per s,
# FUDT downstream fp32 600-1000 per s, ProtT5-XL fp16 30-60 per s, giving 5 to 10.5 h.
# 18 h covers the slow end with margin; a job cut by the limit is resumed by resubmitting.
#SBATCH -p main
#SBATCH --gres=gpu:1
#SBATCH --mem=48g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --job-name=043-iso-embed
#SBATCH --time=18:00:00
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
SCRIPT=experiments/043-expression-proteome-morphology/scripts/compute_isolate_composite.py
echo "host $(hostname)  commit $(git rev-parse --short HEAD)  gpu ${CUDA_VISIBLE_DEVICES:-none}  $(date)"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv
python "$SCRIPT" --stage embed --encoders fudt_species_upstream,calm,fudt_species_downstream
python "$SCRIPT" --stage embed --encoders prot_t5_xl_uniref50
python "$SCRIPT" --stage verify
echo "done $(date)"
