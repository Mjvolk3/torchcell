#!/bin/bash
# experiments/026-metabolism-flux/scripts/gh_cpu_cell_build_verify.slurm
# [[experiments.026-metabolism-flux.physics-gates-2026-09-14]]
#
# VERIFY THE WAVE-9 CELL THROUGH THE PATH IT ACTUALLY RUNS.
#
# Every number the cell was designed on came from probe_warmstart_init.py, which reaches its
# k_cat rules by overwriting layer.kcat_per_h after construction. The cell cannot: it names
# an arm and the table is resolved by parameters.resolve_kcat_table inside 026's
# build_flux_layer. This builds the layer the cell's way and re-reads the five statistics and
# the three watched ceilings against job 1937's row.
#
# PYTHONHASHSEED=0 is not decoration: the resolved table depends on it, 2 of 3,728 values
# moving by up to 0.313 decades, and the script refuses to run without it.
#
#   cd <worktree-root> && sbatch experiments/026-metabolism-flux/scripts/gh_cpu_cell_build_verify.slurm
#
#SBATCH -p main
#SBATCH --mem=24g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nice=1000
#SBATCH --job-name=026-cell-build-verify
#SBATCH --time=01:00:00
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
    experiments/026-metabolism-flux/scripts/probe_cell_build_verify.py --batch 4
