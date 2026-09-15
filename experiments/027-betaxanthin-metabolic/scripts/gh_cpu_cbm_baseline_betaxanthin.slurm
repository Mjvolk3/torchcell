#!/bin/bash
# experiments/027-betaxanthin-metabolic/scripts/gh_cpu_cbm_baseline_betaxanthin.slurm
# [[experiments.027-betaxanthin-metabolic.cpu-baselines-2026-09-14]]
#
# E1 of the metabolic review: the per-genotype constraint-based baseline on the Cachera
# betaxanthin screen, over the 027 pinned test genes. Single-core LP work, no GPU. The
# database rebuild owns this machine right now, so this job is niced to the floor and asks
# for 4 cores and 16G.
#
# PRE-REGISTERED READ. Spearman on the pinned GEM genes with a bootstrap 95 % CI and a
# permutation p; top-10/25/50 precision against the 0.1563 base rate; the fraction of
# genotypes at the modal value beside every correlation.
#   SUCCESS: a readout's CI excludes 0 with the tie fraction below 0.9.
#   KILL:    every readout's CI spans 0 and every tie fraction is above 0.9. Publish the
#            measured null and stop; do not tune the growth fraction, the medium or the
#            objective afterwards.
#SBATCH -p main
#SBATCH --nice=1000
#SBATCH --mem=16g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=027-cbm-btx
#SBATCH --time=03:00:00
#SBATCH --output=/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/kinetics-equilibrator-datasets/experiments/027-betaxanthin-metabolic/slurm/output/%x_%j.out
#SBATCH --error=/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/kinetics-equilibrator-datasets/experiments/027-betaxanthin-metabolic/slurm/output/%x_%j.out

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/kinetics-equilibrator-datasets}"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate torchcell
set -a; . "$PROJECT_ROOT/.env"; set +a
export EXPERIMENT_ROOT="$PROJECT_ROOT/experiments"
cd "$PROJECT_ROOT" || exit 1
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS=ignore
echo "host $(hostname); DATA_ROOT=$DATA_ROOT"

python experiments/027-betaxanthin-metabolic/scripts/cbm_baseline_bx.py \
    --target betaxanthin --primary-condition sc_markers
