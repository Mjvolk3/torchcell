#!/bin/bash
# experiments/026-metabolism-flux/scripts/gh_cpu_transport_tfa.slurm
# [[experiments.026-metabolism-flux.physics-gates-2026-09-14]]
#
# E3 GATE. `transport_delta_g_source` has defaulted to "off" in every 026 and 027 run, so
# every multi-compartment reaction has been priced as if the proton-motive force were free,
# and ATP synthase sits in the violated set of the thermodynamic-feasibility LP at about
# the energy of one proton across the mitochondrial gradient. This cell builds the
# transport table with torchcell.metabolism.transport and re-solves that LP.
#
# Read: does the violated count fall below the transport-off count, and does r_0226 leave
# the list? Kill rule (review, rank 3): if the CPU LP does not improve, the transport term
# is a correctness fix for the writeup, not a Genesis lever, and it does not get a GPU.
#
#   cd <worktree-root> && sbatch experiments/026-metabolism-flux/scripts/gh_cpu_transport_tfa.slurm
#
#SBATCH -p main
#SBATCH --mem=24g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nice=1000
#SBATCH --job-name=026-transport-tfa
#SBATCH --time=02:00:00
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

# The route rides in an environment variable so the shipped table, where the review named
# ATP synthase as an offender, can be checked with the same cell:
#   sbatch --export=ALL,TC_THERMO_ROUTE=shipped .../gh_cpu_transport_tfa.slurm
ROUTE="${TC_THERMO_ROUTE:-union_external}"
~/miniconda3/envs/torchcell/bin/python \
    experiments/026-metabolism-flux/scripts/probe_transport_tfa.py \
    --route "$ROUTE" --out "probe_transport_tfa_${ROUTE}.json"
