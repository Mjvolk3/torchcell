#!/bin/bash
# experiments/027-betaxanthin-metabolic/scripts/gh_cpu_cbm_baseline_amino_acids.slurm
# [[experiments.027-betaxanthin-metabolic.cpu-baselines-2026-09-14]]
#
# E1 of the metabolic review, amino-acid half: the per-genotype constraint-based baseline
# on the Mulleder 2016 intracellular amino-acid panel, over the 027 pinned test genes.
#
# THE STRAIN AND THE MEDIUM ARE NOT CACHERA'S. Mulleder screened the PROTOTROPHIC deletion
# collection, so no marker deletions are applied, and the medium is synthetic minimal
# (FBA_MEDIA["SM"]: YNB vitamins + ammonium sulfate + D-glucose, no amino-acid supplement),
# which is the only medium under which an amino-acid synthesis capacity means anything: on
# a supplemented medium the cell imports what it needs and the drain is uninformative.
#
# The readout is the maximum drain of each of the 19 measured amino acids at 90 % of the
# reference strain's growth, added as a demand reaction on the same cytosolic GEM species
# 026's flux head reads (train_flux.AMINO_ACID_TO_GEM_METABOLITE).
#
# PRE-REGISTERED READ. The mean per-feature Spearman over the 19 columns, matching the
# trainer's `metrics.pinned.mean_spearman`, with a per-column permutation p, a bootstrap
# 95 % CI and the per-column n beside it, because the columns are not measured on the same
# genotypes and a mean over unequal sample sizes is not a mean over one gene set.
#   SUCCESS: a column's CI excludes 0 with its tie fraction below 0.9.
#   KILL:    every column's CI spans 0. Publish the measured null and stop.
#SBATCH -p main
#SBATCH --nice=1000
#SBATCH --mem=16g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=027-cbm-aa
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
    --target amino_acids --primary-condition sm_none
