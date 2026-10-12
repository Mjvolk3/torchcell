#!/bin/bash
# experiments/026-metabolism-flux/scripts/gh_cpu_moma_reference.slurm
# [[experiments.026-metabolism-flux.physics-gates-2026-09-14]]
#
# WOULD A MOMA REFERENCE BE A DIFFERENT REFERENCE?
#
# The per-genotype pFBA table moves 201 of the 939 in-GEM gene sets and changes the init
# statistics to three decimals (job 1973). MOMA is the note's own suggestion for a reference
# that varies per genotype, on the argument that a minimal adjustment from the wild type
# moves whenever the knockout forces any rerouting. This measures the three distances on 30
# of the 201 moved genes BEFORE any table is built, because the argument fails in both
# directions: MOMA on the pFBA knockout solution is the same table, MOMA on the wild type is
# the single-reference head job 1946 already ran.
#
# No QP package is installed (optlang carries GLPK and SCIPY, LP only), so the probe carries
# its own OSQP-style ADMM and refuses to report a distance unless the wild-type round trip
# returns the wild-type flux.
#
#   cd <worktree-root> && sbatch experiments/026-metabolism-flux/scripts/gh_cpu_moma_reference.slurm
#
#SBATCH -p main
#SBATCH --mem=24g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nice=1000
#SBATCH --job-name=026-moma-reference
#SBATCH --time=03:00:00
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
    experiments/026-metabolism-flux/scripts/probe_moma_reference.py --genes 30
