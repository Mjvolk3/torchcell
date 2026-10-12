#!/bin/bash
# experiments/027-betaxanthin-metabolic/scripts/gh_bx_reference_flux_table.slurm
# [[experiments.027-betaxanthin-metabolic.scripts.bx_reference_flux_table]]
#
# The betaxanthin strain's layer-free store (bx_reference_flux_table.py): two pfba solves
# per GEM gene on the cassette model, BY4741 markers deleted, corrected YPD. CPU only.
#
#   sbatch experiments/027-betaxanthin-metabolic/scripts/gh_bx_reference_flux_table.slurm
#
#SBATCH -p main
#SBATCH --job-name=027-bx-store
#SBATCH --cpus-per-task=8
#SBATCH --mem=24g
#SBATCH --time=04:00:00
#SBATCH --output=experiments/027-betaxanthin-metabolic/slurm/output/%x_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONHASHSEED=0
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export PYTHONPATH="$SLURM_SUBMIT_DIR"
echo "worktree $SLURM_SUBMIT_DIR, job $SLURM_JOB_ID, $(date)"
~/miniconda3/envs/torchcell/bin/python \
    experiments/027-betaxanthin-metabolic/scripts/bx_reference_flux_table.py \
    --processes "$SLURM_CPUS_PER_TASK" ${STORE_ARGS:-}
echo "done $(date)"
