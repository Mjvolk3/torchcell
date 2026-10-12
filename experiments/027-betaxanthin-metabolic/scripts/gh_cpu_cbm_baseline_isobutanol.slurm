#!/bin/bash
# experiments/027-betaxanthin-metabolic/scripts/gh_cpu_cbm_baseline_isobutanol.slurm
# [[experiments.027-betaxanthin-metabolic.cpu-baselines-2026-09-14]]
#
# E1 of the metabolic review, isobutanol half: the per-genotype constraint-based baseline
# on the Montano Lopez 2024 biosensor screen, over the 027 pinned test genes.
#
# NO CASSETTE IS NEEDED. Yeast-GEM 9.0.2 already carries isobutanol (s_0932/3/4),
# isobutyraldehyde (s_0937/8/9) and r_1866, an isobutanol exchange, so the objective is set
# rather than built. The strain is BY4741 ura3D0 with the Leu3p/LEU1-yEGFP reporter
# integrated at HIS3, on SC, so the four BY4741 marker deletions are applied.
#
# The biosensor does not report isobutanol; it reports alpha-ketoisovalerate through Leu3p,
# whose ligand is 2-isopropylmalate. So the production capacity of s_0162 (cytosolic) and
# s_0164 (mitochondrial) is read out beside r_1866 rather than instead of it.
#
# PRE-REGISTERED READ. Spearman against log10 of the released fold change on the pinned
# GEM genes, with a bootstrap 95 % CI and a permutation p; top-10/25/50 precision against
# the 0.1563 base rate; the fraction of genotypes at the modal value beside every
# correlation.
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
#SBATCH --job-name=027-cbm-ibu
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

# THE MARKER-FREE CONTROL IS NOT RUN, and the reason is measured rather than assumed. Job
# 1903 completed `sc_markers` in 8.3 min and then stalled inside `sc_none` at genotype
# 300 of 581, hitting the 3 h wall with no further progress line: one LP in that condition
# does not return. The primary condition is what the pre-registered read needs, so the run
# is narrowed to it and the stall is recorded as an open item rather than worked around.
python experiments/027-betaxanthin-metabolic/scripts/cbm_baseline_bx.py \
    --target isobutanol --primary-condition sc_markers --conditions sc_markers
