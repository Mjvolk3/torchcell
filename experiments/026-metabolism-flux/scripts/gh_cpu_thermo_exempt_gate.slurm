#!/bin/bash
# experiments/026-metabolism-flux/scripts/gh_cpu_thermo_exempt_gate.slurm
# [[experiments.026-metabolism-flux.physics-gates-2026-09-14]]
#
# E2 GATE. Exempting the reactions no concentration can rescue is worth GPU time only if
# the thermodynamic infeasibility is CONCENTRATED. probe_tfa_lp.py measures a total L1
# slack of thousands of kJ/mol under union_external with 47 reactions violated; this cell
# drops those and re-solves, then falls through to a greedy minimum-cardinality removal if
# the residual misses the ~10 kJ gate.
#
# Kill rule (review, rank 2): if no small removal set drops the slack below about 10 kJ,
# the infeasibility is distributed and E2 is dead before any GPU runs.
#
#   cd <worktree-root> && sbatch experiments/026-metabolism-flux/scripts/gh_cpu_thermo_exempt_gate.slurm
#
#SBATCH -p main
#SBATCH --mem=24g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nice=1000
#SBATCH --job-name=026-thermo-exempt-gate
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

~/miniconda3/envs/torchcell/bin/python \
    experiments/026-metabolism-flux/scripts/thermo_exempt_gate.py --route union_external
