#!/bin/bash
# experiments/026-metabolism-flux/scripts/gh_cpu_genotype_reference_init.slurm
# [[experiments.026-metabolism-flux.physics-gates-2026-09-14]]
#
# THE INIT CHECK ON REAL DATASET GENOTYPES, with and without the per-genotype reference.
#
# Every earlier init check forwarded four WILD TYPES, which is why none of them saw what the
# job-1946 read shows: the training-batch active-row balance max was already about 1.0 at
# epoch 0 while the wild-type probe read 0.03, so the deletion genotypes fail the rule from
# the start. This loads 32 real genotypes through the trainer's own loader and reports the
# five statistics, the across-genotype flux sd, and the fraction of genotypes whose reference
# actually differs from the wild type.
#
#   cd <worktree-root> && sbatch experiments/026-metabolism-flux/scripts/gh_cpu_genotype_reference_init.slurm
#
#SBATCH -p main
#SBATCH --mem=24g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nice=1000
#SBATCH --job-name=026-genoref-init
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
    experiments/026-metabolism-flux/scripts/probe_genotype_reference_init.py --genotypes 32
