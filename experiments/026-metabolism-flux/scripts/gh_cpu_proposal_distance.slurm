#!/bin/bash
# experiments/026-metabolism-flux/scripts/gh_cpu_proposal_distance.slurm
# [[experiments.026-metabolism-flux.physics-gates-2026-09-14]]
#
# X0a: HOW FAR IS A TRAINED HEAD'S PROPOSAL FROM FEASIBLE?
#
# The case for a warm-started residual head is that the PROPOSAL, not the projection, is
# what fails the turnover-thresholded balance rule. probe_warmstart.py establishes the
# dependence on proposal quality synthetically: a proposal whose projection displacement is
# 0.008 of its own norm clears the rule at 0.0012, and a uniform random point of the box,
# displacement 0.95, sits at 2.0. Neither is a checkpoint.
#
# This cell rebuilds job 1784 (density 2.0, seed 2700, fluxforce_k100_boxforce), forwards 32
# genotypes and reports ||v_pre - P(v_pre)|| / ||v_pre|| off the layer's own
# v_pre_projection, then places it in that table.
#
# It GATES E1: a checkpoint near the polytope makes the warm start nearly free, and one an
# O(1) distance away makes it the experiment.
#
#   cd <worktree-root> && sbatch experiments/026-metabolism-flux/scripts/gh_cpu_proposal_distance.slurm
#
#SBATCH -p main
#SBATCH --mem=24g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nice=1000
#SBATCH --job-name=026-proposal-distance
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
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

PY=~/miniconda3/envs/torchcell/bin/python
echo "host $(hostname); DATA_ROOT=$DATA_ROOT"

$PY experiments/026-metabolism-flux/scripts/probe_trained_proposal_distance.py \
    --result wave8_dens_d2p0_boxforce_k100_e80_s2700.json --genotypes 32
