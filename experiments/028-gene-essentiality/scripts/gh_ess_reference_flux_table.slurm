#!/bin/bash
# experiments/028-gene-essentiality/scripts/gh_ess_reference_flux_table.slurm
# [[experiments.028-gene-essentiality.scripts.ess_reference_flux_table]]
#
# One pFBA store (ess_reference_flux_table.py) under slurm. STORE_ARGS carries the
# script's arguments; the portable form over every GEM gene on one medium is
#
#   STORE_ARGS="--medium SM --genes gem --out-tag gem" \
#     sbatch experiments/028-gene-essentiality/scripts/gh_ess_reference_flux_table.slurm
#
# which writes ess_solve_encodings_SM_gem.npz (and the reference table) under
# $DATA_ROOT/data/torchcell/reference_flux. CPU only; the deletion solves are serial
# (about 0.5 s each) and the FVA uses 8 processes.
#
#SBATCH -p main
#SBATCH --job-name=028-pfba-store
#SBATCH --cpus-per-task=8
#SBATCH --mem=24g
#SBATCH --time=04:00:00
#SBATCH --output=experiments/028-gene-essentiality/slurm/output/%x_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONHASHSEED=0
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export PYTHONPATH="$SLURM_SUBMIT_DIR"
echo "worktree $SLURM_SUBMIT_DIR, job $SLURM_JOB_ID, args ${STORE_ARGS:-}, $(date)"
~/miniconda3/envs/torchcell/bin/python \
    experiments/028-gene-essentiality/scripts/ess_reference_flux_table.py ${STORE_ARGS:-}
echo "done $(date)"
