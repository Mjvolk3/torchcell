#!/bin/bash
# experiments/026-metabolism-flux/scripts/gh_cpu_thermo_exempt_gate_media.slurm
# [[experiments.026-metabolism-flux.physics-gates-2026-09-14]]
#
# THE MEDIUM-CONSISTENT EXEMPTION LIST AND CONCENTRATION WITNESS.
#
# Both artifacts are functions of the flux they are priced at, and both shipped ones were
# solved on the SHIPPED boundary against the cobra FBA vertex. The 2x2 (job 1974) measures
# that the MEDIUM is what carries the warm start's infeasibility, so a layer built on YPD is
# being tightened by a list and a concentration vector solved for a different cell. This
# re-solves both on each medium against the flux layer's OWN pFBA reference, so the artifacts
# license the exact vector the layer clips into its box.
#
# Two invocations of one script; the second is the like-for-like shipped-bounds control, so
# the medium is the only difference between the two pairs.
#
#   cd <worktree-root> && sbatch experiments/026-metabolism-flux/scripts/gh_cpu_thermo_exempt_gate_media.slurm
#
#SBATCH -p main
#SBATCH --mem=24g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nice=1000
#SBATCH --job-name=026-thermo-exempt-media
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
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

set -e

~/miniconda3/envs/torchcell/bin/python \
    experiments/026-metabolism-flux/scripts/thermo_exempt_gate.py \
    --route union_external --medium YPD --flux pfba_layer \
    --out thermo_exempt_candidates_ypd.json \
    --witness-out thermo_consistent_log_c_ypd.json \
    --centered-out thermo_consistent_log_c_ypd_centered.json

~/miniconda3/envs/torchcell/bin/python \
    experiments/026-metabolism-flux/scripts/thermo_exempt_gate.py \
    --route union_external --flux pfba_layer \
    --out thermo_exempt_candidates_minimal_pfba.json \
    --witness-out thermo_consistent_log_c_minimal_pfba.json \
    --centered-out thermo_consistent_log_c_minimal_pfba_centered.json
