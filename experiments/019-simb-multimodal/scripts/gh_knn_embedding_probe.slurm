#!/bin/bash
# experiments/019-simb-multimodal/scripts/gh_knn_embedding_probe.slurm
# GilaHyper launcher for the parameter-free kNN embedding probe.
#
# Answers whether embedding proximity implies phenotype similarity, independent of the
# transformer and independent of the graph channel. Cheap: no training, no backprop --
# a cosine similarity matrix (534 x 4074) and a weighted gather per k. Minutes, not hours.
#
# Runs on 1 GPU purely to match the normal experiment convention and to keep the embedding
# loaders on their usual device path; the probe itself is numpy on CPU. GPU 3 is the one
# the concurrent expr sweep (gh_expr_embed_005.slurm) deliberately leaves free.
#
#   cd <worktree-root> && sbatch experiments/019-simb-multimodal/scripts/gh_knn_embedding_probe.slurm
#
#SBATCH -p main
#SBATCH --mem=64g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --job-name=019-knn-probe
#SBATCH --time=04:00:00
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/scratch/projects/torchcell-scratch/experiments/019-simb-multimodal/slurm/output/%x_%j.out
#SBATCH --error=/scratch/projects/torchcell-scratch/experiments/019-simb-multimodal/slurm/output/%x_%j.out

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-/home/michaelvolk/Documents/projects/torchcell}"
mkdir -p /scratch/projects/torchcell-scratch/experiments/019-simb-multimodal/slurm/output

source ~/miniconda3/etc/profile.d/conda.sh
conda activate torchcell
cd "$PROJECT_ROOT" || exit 1

export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS=ignore

python "$PROJECT_ROOT/experiments/019-simb-multimodal/scripts/knn_embedding_probe.py"
