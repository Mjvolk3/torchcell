#!/bin/bash
# experiments/026-metabolism-flux/scripts/gh_cpu_fva_kcat_sweep_b.slurm
# [[experiments.026-metabolism-flux.kcat-uncertainty-2026-09-14]]
#
# E1 SWEEP, SHARD 1 of 2. See shard 0 for the design; the shards split the same 17-cell
# list and write separate result files.
#
# CPU ONLY, NICED BEHIND THE DATABASE REBUILD. The rb_*, verify_* and kg_* chain owns this
# machine, so this job takes no GPU, at most 8 cores and at most 24G, and sits at
# --nice=1000 behind everything already queued.
#
# PYTHONHASHSEED=0 pins the resolver's tie-breaking: with a free seed two processes resolve
# 2 of the 3,728 units differently by up to 0.313 decades, and the shards of this sweep
# would then be compared against each other on slightly different tables.
#SBATCH -p main
#SBATCH --nice=1000
#SBATCH --mem=24g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=026-fva-kcat-sweep-b
#SBATCH --time=10:00:00
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
echo "host $(hostname); DATA_ROOT=$DATA_ROOT; cpus $SLURM_CPUS_PER_TASK"

python experiments/026-metabolism-flux/scripts/fva_kcat_uncertainty.py \
    --mode sweep --medium sbml_default --draws 200 --processes 8 \
    --shard 1 --n-shards 2 --out fva_kcat_sweep_b
