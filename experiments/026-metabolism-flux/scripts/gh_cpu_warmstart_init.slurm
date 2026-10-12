#!/bin/bash
# experiments/026-metabolism-flux/scripts/gh_cpu_warmstart_init.slurm
# [[experiments.026-metabolism-flux.physics-gates-2026-09-14]]
#
# E1 INIT CHECK. Build the warm-started residual head through 026's own build_flux_layer on
# the real yeast-GEM, untrained, and read the four numbers the arm is judged by:
# balance_active_max, balance_active_p95, flux_active_frac and the active-row count, beside
# the fluxforce_k100_boxforce control built identically at the same seed.
#
# The target is the rel-0.01 row of probe_warmstart.json (0.00118 / 0.00038 / 0.21 / 82),
# which was measured with no genotype box on it. The genotype box carries the enzyme
# capacity bound, and a reference clipped into a box narrower than itself is no longer
# balanced however good the reference was, so this is the check that says whether the GPU
# cells are worth submitting.
#
#   cd <worktree-root> && sbatch experiments/026-metabolism-flux/scripts/gh_cpu_warmstart_init.slurm
#
#SBATCH -p main
#SBATCH --mem=24g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nice=1000
#SBATCH --job-name=026-warmstart-init
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
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# TC_INIT_ONLY selects a subset of the settings by name prefix, so the attribution run
# does not rebuild the whole grid:
#   sbatch --export=ALL,TC_INIT_ONLY=attrib,TC_INIT_OUT=probe_warmstart_attrib.json ...
ONLY="${TC_INIT_ONLY:-}"
OUT="${TC_INIT_OUT:-probe_warmstart_init.json}"
~/miniconda3/envs/torchcell/bin/python \
    experiments/026-metabolism-flux/scripts/probe_warmstart_init.py \
    --batch 4 --only "$ONLY" --out "$OUT"
