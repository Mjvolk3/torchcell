#!/bin/bash
#SBATCH -p main
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --mem=48g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=019-cellpose-seg
#SBATCH --time=2:00:00
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type=END
#SBATCH --output=/home/michaelvolk/Documents/projects/torchcell/experiments/019-echo-crispr-array/slurm/output/%x_%j.out
#SBATCH --error=/home/michaelvolk/Documents/projects/torchcell/experiments/019-echo-crispr-array/slurm/output/%x_%j.out

# GilaHyper GPU job: Cellpose-SAM colony segmentation for the run-2 (2026-07-17)
# ECHO CRISPR plates. One RTX 6000 Ada is plenty (cpsam eval ~1 min/full-res plate,
# 6 plates). --gres=gpu:rtx6000:1 pins one MIG/GPU; do NOT hardcode
# CUDA_VISIBLE_DEVICES -- slurm sets it from the gres allocation.

# Change to the project root (required for the script's relative EXP_DIR paths)
cd /home/michaelvolk/Documents/projects/torchcell || exit 1

# Activate conda environment directly
source ~/miniconda3/bin/activate
conda activate torchcell

# Add project to PYTHONPATH
export PYTHONPATH="/home/michaelvolk/Documents/projects/torchcell:$PYTHONPATH"
export PYTHONUNBUFFERED=1

# cpsam weights auto-download to ~/.cellpose/models on first use (~1.15 GB); on an
# offline node pre-sync that dir from a networked run.
~/miniconda3/envs/torchcell/bin/python \
  experiments/019-echo-crispr-array/scripts/run2_cellpose_segmentation.py
