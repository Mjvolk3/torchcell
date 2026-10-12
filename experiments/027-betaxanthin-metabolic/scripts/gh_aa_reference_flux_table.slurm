#!/bin/bash
# experiments/027-betaxanthin-metabolic/scripts/gh_aa_reference_flux_table.slurm
# [[experiments.027-betaxanthin-metabolic.scripts.aa_reference_flux_table]]
#
# The amino-acid screen's layer-free store (aa_reference_flux_table.py): one pfba growth solve
# and 19 amino-acid maxima per GEM gene, prototroph on synthetic minimal medium. CPU only.
#
#   sbatch experiments/027-betaxanthin-metabolic/scripts/gh_aa_reference_flux_table.slurm
#
#SBATCH -p main
#SBATCH --job-name=027-aa-store
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
    experiments/027-betaxanthin-metabolic/scripts/aa_reference_flux_table.py \
    --processes "$SLURM_CPUS_PER_TASK" ${STORE_ARGS:-}
echo "done $(date)"
