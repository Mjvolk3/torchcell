#!/bin/bash
# experiments/027-betaxanthin-metabolic/scripts/gh_cpu_linear_baselines_amino_acids.slurm
# [[experiments.027-betaxanthin-metabolic.cpu-baselines-2026-09-14]]
#
# E3 of the metabolic review: 019's B0-B3 linear ladder on the 027 pinned split, the
# Mulleder 19 amino acids.
#
# B2 KEEPS ITS BILINEAR FORM HERE. The target is 19 columns, so the gene basis is real and
# the rank grid is swept up to 19; on the scalar targets the same model collapses to plain
# ridge. The panel is not measured on every genotype, so a missing cell is set to the fit
# column mean for the SVD and the ridge solve and is never scored.
#
# WHY IT MATTERS. If a ridge on a protein-language-model embedding of the deleted gene
# reaches a non-trivial fraction of the CGT pooled arm's mean per-feature Spearman, the bar
# the metabolic module has to clear is set by GENE IDENTITY rather than by flux.
#
# CONTROL. B0, the train mean, must score exactly 0 on per-feature Pearson and its NMSE must
# land in (0.9, 1.15); the script raises otherwise. B1 is the zero predictor.
#
# PRE-REGISTERED READ. Test Spearman AND top-50 precision at the validation-selected
# hyperparameter, per embedding set, per model, with a permutation p and a bootstrap CI on
# the selected cell.
#   SUCCESS: any linear cell's CI excludes 0.
#   KILL:    every cell sits inside the +/- 0.04 null width. Report the null; do not go
#            hunting for a fifth embedding.
#SBATCH -p main
#SBATCH --nice=1000
#SBATCH --mem=24g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=027-lin-aa
#SBATCH --time=04:00:00
#SBATCH --output=/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/kinetics-equilibrator-datasets/experiments/027-betaxanthin-metabolic/slurm/output/%x_%j.out
#SBATCH --error=/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/kinetics-equilibrator-datasets/experiments/027-betaxanthin-metabolic/slurm/output/%x_%j.out

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/kinetics-equilibrator-datasets}"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate torchcell
set -a; . "$PROJECT_ROOT/.env"; set +a
export EXPERIMENT_ROOT="$PROJECT_ROOT/experiments"
cd "$PROJECT_ROOT" || exit 1
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS=ignore
export CUDA_VISIBLE_DEVICES=""
echo "host $(hostname); DATA_ROOT=$DATA_ROOT"

python experiments/027-betaxanthin-metabolic/scripts/linear_baselines_bx.py \
    --target amino_acids
