#!/bin/bash
# experiments/026-metabolism-flux/scripts/gh_cpu_reference_flux_table.slurm
# [[experiments.026-metabolism-flux.physics-gates-2026-09-14]]
#
# BUILD ONE REFERENCE FLUX PER GENOTYPE.
#
# Job 1946 measured what a single wild-type reference costs: the proposal is nearly the same
# vector for every genotype, the head reading it is close to a constant predictor, and the
# only channel the data term had for genotype information was alpha, which it grew from
# 0.0099 to 0.087 at the cost of the balance rule. This solves pFBA once per DISTINCT
# deleted-gene set in the dataset, with the genes knocked out through the GPR.
#
# Minutes of CPU, paid once: the parameter review measured 0.06 s per knockout FBA.
#
#   cd <worktree-root> && sbatch experiments/026-metabolism-flux/scripts/gh_cpu_reference_flux_table.slurm
#
#SBATCH -p main
#SBATCH --mem=24g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nice=1000
#SBATCH --job-name=026-reference-flux-table
#SBATCH --time=04:00:00
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

# The medium and the output name ride in the environment so one cell builds both tables:
#   sbatch --export=ALL,TC_MEDIUM=YPD,TC_TABLE_OUT=reference_flux_table_ypd.npz ...
#   sbatch --export=ALL,TC_MEDIUM=none,TC_TABLE_OUT=reference_flux_table_minimal.npz ...
MEDIUM="${TC_MEDIUM:-YPD}"
TABLE_OUT="${TC_TABLE_OUT:-reference_flux_table_ypd.npz}"
~/miniconda3/envs/torchcell/bin/python \
    experiments/026-metabolism-flux/scripts/probe_reference_flux_table.py \
    --medium "$MEDIUM" --out "$TABLE_OUT"
