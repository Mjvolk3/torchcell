#!/bin/bash
# experiments/025-solid-growth/scripts/gh_wt_gene_embedding_clusters.slurm
# [[experiments.025-solid-growth.scripts.wt_gene_embedding_clusters]]
#
# GilaHyper CPU job for wt_gene_embedding_clusters.py: load the S3 closure checkpoint
# (8fuwsk48, epoch 43), re-score the 37,673 validation triples as a load check, extract the
# gene table and the wild-type encoder output, then cluster + GO enrichment + panels.
# CPU only: the encoder runs at batch 1 over 6,608 tokens, and no GPU card is needed.
#
#   PROJECT_ROOT=<worktree> sbatch -J 025-wtclust-w1-all \
#       experiments/025-solid-growth/scripts/gh_wt_gene_embedding_clusters.slurm [--analyze-only]
#SBATCH -p main
#SBATCH --mem=48g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=025-wtclust
#SBATCH --time=01:30:00
#SBATCH --output=/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/025-graph-reg-sweep-readout/experiments/025-solid-growth/slurm/output/%x_%j.out
#SBATCH --error=/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/025-graph-reg-sweep-readout/experiments/025-solid-growth/slurm/output/%x_%j.out

PROJECT_ROOT="${PROJECT_ROOT:-/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/025-graph-reg-sweep-readout}"
cd "$PROJECT_ROOT" || exit 1
export PYTHONPATH="$PROJECT_ROOT"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "job $SLURM_JOB_ID on $(hostname), commit $(git -C "$PROJECT_ROOT" rev-parse --short HEAD)"
~/miniconda3/envs/torchcell/bin/python experiments/025-solid-growth/scripts/wt_gene_embedding_clusters.py "$@"
