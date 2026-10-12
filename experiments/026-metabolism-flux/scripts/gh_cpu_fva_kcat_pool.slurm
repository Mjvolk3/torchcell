#!/bin/bash
# experiments/026-metabolism-flux/scripts/gh_cpu_fva_kcat_pool.slurm
# [[experiments.026-metabolism-flux.kcat-uncertainty-2026-09-14]]
#
# E6. The protein pool swept over p_avail in {0.125, 0.25, 0.35} g/gDW crossed with a
# saturation factor in {0.5, 1.0}, under two enforcements that are not the same constraint:
# a GECKO-light LP pool that allocates protein optimally, and the equal-molar abundance the
# flux layer prior uses, which allocates nothing. The two bracket the growth a pool permits,
# and quoting either alone would overstate the case.
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
#SBATCH --mem=16g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --job-name=026-fva-kcat-pool
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
export PYTHONHASHSEED=0
echo "host $(hostname); DATA_ROOT=$DATA_ROOT; cpus $SLURM_CPUS_PER_TASK"

python experiments/026-metabolism-flux/scripts/fva_kcat_uncertainty.py \
    --mode pool --medium sbml_default --out fva_kcat_pool
