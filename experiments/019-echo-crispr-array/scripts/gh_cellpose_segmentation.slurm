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
#SBATCH --output=/home/michaelvolk/Documents/projects/torchcell.worktrees/paper/figures-fig1/experiments/019-echo-crispr-array/slurm/output/%x_%j.out
#SBATCH --error=/home/michaelvolk/Documents/projects/torchcell.worktrees/paper/figures-fig1/experiments/019-echo-crispr-array/slurm/output/%x_%j.out

# GilaHyper GPU job: the full run-2 Cellpose scorer (segment -> register -> normalize
# -> per-strain fitness + Cellpose-vs-classical), tuned recipe + invalidation model.
# One RTX 6000 Ada; --gres=gpu:rtx6000:1 pins one GPU (do NOT hardcode
# CUDA_VISIBLE_DEVICES). NOTE: worktree-scoped paths (branch paper/figures-fig1);
# repoint to the primary checkout when this lands.

WT=/home/michaelvolk/Documents/projects/torchcell.worktrees/paper/figures-fig1
cd "$WT" || exit 1

source ~/miniconda3/bin/activate
conda activate torchcell
export PYTHONPATH="$WT:$PYTHONPATH"
export PYTHONUNBUFFERED=1

~/miniconda3/envs/torchcell/bin/python \
  experiments/019-echo-crispr-array/scripts/run2_cellpose_segmentation.py
