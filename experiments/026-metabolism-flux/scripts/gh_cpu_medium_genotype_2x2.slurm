#!/bin/bash
# experiments/026-metabolism-flux/scripts/gh_cpu_medium_genotype_2x2.slurm
# [[experiments.026-metabolism-flux.physics-gates-2026-09-14]]
#
# THE 2x2 THAT SEPARATES THE MEDIUM FROM THE GENOTYPES.
#
# The warm-start init check reads balance_active_max 0.0306 on four WILD TYPES on the SHIPPED
# bounds (job 1937) and 1.5898 on 32 real DELETION genotypes on YPD (job 1973). Two things
# changed at once. This crosses them at one seed with one wild-type reference per medium, and
# carries the clip-cost decomposition per cell so a wider or narrower box says where it bites.
#
#   cd <worktree-root> && sbatch experiments/026-metabolism-flux/scripts/gh_cpu_medium_genotype_2x2.slurm
#
#SBATCH -p main
#SBATCH --mem=24g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nice=1000
#SBATCH --job-name=026-med-geno-2x2
#SBATCH --time=01:30:00
#SBATCH --output=/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/kinetics-equilibrator-datasets/experiments/026-metabolism-flux/slurm/output/%x_%j.out
#SBATCH --error=/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/kinetics-equilibrator-datasets/experiments/026-metabolism-flux/slurm/output/%x_%j.out

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
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

~/miniconda3/envs/torchcell/bin/python \
    experiments/026-metabolism-flux/scripts/probe_medium_genotype_2x2.py \
    --genotypes 32 ${TC_2X2_ARGS}
