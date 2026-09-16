#!/bin/bash
# experiments/027-betaxanthin-metabolic/scripts/gh_cpu_linear_baselines_isobutanol.slurm
# [[experiments.027-betaxanthin-metabolic.cpu-baselines-2026-09-14]]
#
# E3 of the metabolic review: 019's B0-B3 linear ladder on the 027 pinned split, isobutanol.
#
# A DIFFERENT BUILD, AND THE MEASUREMENT THAT PICKED IT. `fig6_pigment_transfer`, the build
# the 027 arms train on, carries no isobutanol column. `fig6_build` does: MEASURED, 4,244 of
# its records hold a width-20 `metabolite_level` (the Mulleder 19 plus isobutanol, merged by
# the deletion-keyed aggregator) and 189 hold an isobutanol-only record, so 4,433 genotypes
# carry the Lopez biosensor readout. The pinned test list is the same Merzbacher 639, so the
# partition is comparable even though the phenotype is not the one the arms train on.
#
# NO 027 ARM PREDICTS ISOBUTANOL, so this is a baseline with no model beside it yet. It says
# what a linear model on gene identity reaches on the target, which is what a future arm
# would have to clear.
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
#SBATCH --mem=16g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=027-lin-ibu
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
    --target isobutanol
