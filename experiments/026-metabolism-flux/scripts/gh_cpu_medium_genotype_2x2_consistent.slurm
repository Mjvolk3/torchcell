#!/bin/bash
# experiments/026-metabolism-flux/scripts/gh_cpu_medium_genotype_2x2_consistent.slurm
# [[experiments.026-metabolism-flux.physics-gates-2026-09-14]]
#
# THE 2x2 AGAIN, WITH MEDIUM-CONSISTENT THERMODYNAMIC ARTIFACTS.
#
# Job 1974 measures that the MEDIUM carries the warm start's infeasibility, and it ran every
# cell against an exemption list and a concentration witness solved on the SHIPPED boundary
# against the cobra FBA vertex. Job 1975 re-solved both per medium against the flux layer's
# own pFBA reference (YPD: 98 reactions exempted, residual slack 0.000 kJ over 302
# constraints; shipped bounds: 42 exempted, 0.000 kJ over 360). This rerun gives each medium
# its own pair, so the only thing crossing a medium boundary is the medium.
#
# The GEM-universe control is on: every init check before job 1974 initialized the layer's
# random heads over the GEM's ~1.1k gene list while a real cell passes the transformer's
# 6,607-token universe, and that is a third axis the 2x2 would otherwise absorb.
#
#   cd <worktree-root> && sbatch experiments/026-metabolism-flux/scripts/gh_cpu_medium_genotype_2x2_consistent.slurm
#
#SBATCH -p main
#SBATCH --mem=24g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nice=1000
#SBATCH --job-name=026-med-geno-2x2c
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
    --genotypes 32 --gem-universe-control \
    --exempt-by-medium '{"YPD": "thermo_exempt_candidates_ypd.json", "none": "thermo_exempt_candidates_minimal_pfba.json"}' \
    --log-c-by-medium '{"YPD": "thermo_consistent_log_c_ypd.json", "none": "thermo_consistent_log_c_minimal_pfba.json"}' \
    --out probe_medium_genotype_2x2_consistent.json
