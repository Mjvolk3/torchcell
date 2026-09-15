#!/bin/bash
# experiments/026-metabolism-flux/scripts/gh_cpu_kcat_sigma_table.slurm
# [[experiments.026-metabolism-flux.kcat-uncertainty-2026-09-14]]
#
# E3. Recompute the four rungs of the k_cat width ladder from the Open Enzyme Database and
# from the 58 measured yeast pairs, assert each against the constant that ships, resolve the
# table with and without the width, and compare the four shipped fields element-wise.
#
# CPU ONLY, NICED BEHIND THE DATABASE REBUILD. The dataset-rebuild chain (rb_*, verify_*,
# kg_*) owns this machine right now, so every job in this block runs at --nice=1000 with at
# most 8 cores and 24G and no GPU.
#
# PYTHONHASHSEED=0 is not decoration: two processes with a free seed resolve 2 of the 3,728
# units differently, by up to 0.313 decades, because the resolver breaks a tie through a
# hash-ordered container.
#SBATCH -p main
#SBATCH --nice=1000
#SBATCH --mem=16g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --job-name=026-kcat-sigma-table
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
export PYTHONHASHSEED=0
echo "host $(hostname); DATA_ROOT=$DATA_ROOT"

python experiments/026-metabolism-flux/scripts/kcat_sigma_table.py

python -m pytest tests/torchcell/metabolism/test_parameters.py \
    tests/torchcell/metabolism/test_kinetics_pipeline.py \
    tests/torchcell/metabolism/test_enzyme_kinetics.py -q

python -m pytest tests/torchcell/metabolism/test_kcat_sigma.py -q
