#!/bin/bash
#SBATCH -p main
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --mem=32g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --job-name=019-cellpose-recipe
#SBATCH --time=1:00:00
#SBATCH --mail-type=NONE
#SBATCH --output=/home/michaelvolk/Documents/projects/torchcell.worktrees/paper/figures-fig1/experiments/019-echo-crispr-array/slurm/output/%x_%j.out
#SBATCH --error=/home/michaelvolk/Documents/projects/torchcell.worktrees/paper/figures-fig1/experiments/019-echo-crispr-array/slurm/output/%x_%j.out

# Generic GPU wrapper for a single Cellpose recipe run. Pass the cellpose_recipe.py
# args through sbatch, e.g.:
#   sbatch --job-name=019-chase-clahe02cp4 gh_cellpose_recipe.slurm \
#          --contrast clahe --clahe_clip 0.02 --cellprob -4 \
#          --conditions P1_t50,P1_t72 --tag chase_clahe02_cp4
# One RTX 6000 Ada per job; the 4-GPU node runs up to 4 of these concurrently.
# NOTE: worktree-scoped paths (branch paper/figures-fig1); repoint to the primary
# checkout when this lands.

WT=/home/michaelvolk/Documents/projects/torchcell.worktrees/paper/figures-fig1
cd "$WT" || exit 1

source ~/miniconda3/bin/activate
conda activate torchcell
export PYTHONPATH="$WT:$PYTHONPATH"
export PYTHONUNBUFFERED=1

~/miniconda3/envs/torchcell/bin/python \
  experiments/019-echo-crispr-array/scripts/cellpose_recipe.py "$@"
