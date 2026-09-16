#!/bin/bash
# experiments/027-betaxanthin-metabolic/scripts/gh_wave9_ibprec_e200_s2700.slurm
#
# DRIVER 2026-09-14 20:05: the mechanism test for the flat ibprec cell (job 1928, run
# s4z0ur7o: data loss flat for 195 epochs, the 2-isopropylmalate pool constant across
# genotypes although it varies by 0.07 decades at init). Leading hypothesis (untested): the
# thermodynamic and dissipation terms at constraint scale 0.1 drive log c genotype
# independent and two supervised rows of 2,806 cannot resist. This cell holds every flag of
# job 1928 and sets the constraint scale to 0.0. Read: does the pool keep its spread and does
# the data loss move; pinned Spearman against ibpool s2700 (0.1278) is secondary.
# [[experiments.027-betaxanthin-metabolic.wave9-isobutanol]]
#
# WAVE 9 BLOCK 1: the isobutanol three-way, `ibprec`, seed 2700.
#
# THIS CELL IS THE CONCENTRATION ARM, and it is the PRIMARY one-anchor form. The
# prediction is an affine of the layer's own learned whole-cell concentration of
# alpha-isopropylmalate, summed over its cytosolic and mitochondrial species weighted by
# fractional volume. One anchor, because that is literally the metabolite Leu3p binds;
# the three-anchor form is the secondary and is not run first. The affine is required
# because the target is a fluorescence fold change and no unit-free map takes log10
# molar to it, and it is still a mechanistic tie because the only thing it may read is
# the named pool's concentration.
#
# THE TARGET. The Montano Lopez 2024 genome-wide biosensor screen: GFP expressed from the
# LEU1 promoter under Leu3p, so the number is a fold change in the intracellular pool of
# alpha-isopropylmalate, which yeast-GEM 9.0.2 carries as s_0162 [c] and s_0164 [m]. It is
# the only one of the module's three targets whose measured quantity is a single NAMED GEM
# metabolite pool, and the only one whose product has a native secretion exchange (r_1866),
# so a product-flux arm and a precursor-pool arm can be compared on ONE phenotype.
#
# THE LABEL IS THE BEST CONDITIONED OF THE THREE, measured rather than assumed: across the
# 222 genes measured in both the n=1 screen and the independent n=3 re-screen, log10 FC
# Pearson is 0.915, giving a reliability of roughly 0.58 to 0.77 on the genome-wide labels
# and a Pearson ceiling near 0.76 to 0.88. The amino-acid ceiling is 0.17 and betaxanthin's
# per-strain replicate structure is not released.
#
# THE DATASET IS fig6_build, NOT fig6_pigment_transfer, because the screen is dropped by
# name from the pigment query. It is the July 22 build of 13,948 aggregated genotypes and is
# opened read-only; nothing is rebuilt.
#
# THE TARGET IS COLUMN 8 OF A WIDTH-20 BLOCK, not a width-1 group. MeanExperimentDeduplicator
# merged the screen into the Mulleder record wherever both were measured, so 4,244 genotypes
# carry one key-sorted metabolite_level group of the 19 amino acids plus isobutanol, while
# the width-1 groups in that build are 4,669 betaxanthin records against 189 isobutanol-only
# ones. A width-1 isobutanol head would have read betaxanthin colony color on 96 percent of
# the rows it matched; 026's extract_targets now refuses a genotype with two width-1 groups
# under one label, and this head is decoded by column instead.
#
# THE VALIDATED PANEL IS HELD BACK. The 224-strain n=3 re-screen is the external direction
# check, and the deduplicator averaged its value into the screen's on 208 of the width-20
# blocks, so every genotype deleting a gene that panel measured is masked out of the loss,
# the validation curve and the test score. What is left is 504 of the 639 pinned Merzbacher
# test genes and 3,353 pool genes.
#
# THE TARGET SPACE IS log10 OF THE FOLD CHANGE. The FC distribution is multiplicative (log10
# sd 0.213 over the screen, minimum 0.0386, maximum 7.47), so a Gaussian CRPS is well posed
# there and not in linear FC. Spearman is invariant to the change, so the primary endpoint is
# directly comparable with every other 027 arm.
#
# THE RECIPE is the wave-7 one, copied from gh_wave8_aa_nodiss_e200_s2700.slurm: 200 epochs,
# CRPS, the 019 graph regularizer, the predicted-kcat tier, the verified reaction-energy
# table, constraint scale 0.1 over a 20-epoch warmup, concentration window raised to 1 M. All
# three arms share it, so the three-way contrast is on the readout alone.
#
# PRE-REGISTERED READ, before a single cell runs: `ibprec - ibpool` and `ibscalar - ibpool`,
# paired by seed, against a noise floor the three `ibpool` seeds establish themselves; plus
# the external check on the 224 re-screened strains. SUCCESS/KILL: `ibprec - ibpool` above
# +0.02 with all three seeds positive means the module transfers to a second, better
# conditioned target and the concentration mechanism is not betaxanthin-specific. All three
# arms within 0.01 means the target is not sensitive to the module, and the honest report is
# that the biosensor reads chromatin and mitochondrial state (the paper's own top hits are
# histone modifiers: spt10, dep1, sap30, gcn5, swi4, swi6) rather than pathway flux. If
# `ibscalar` wins, that is the first evidence anywhere that a flux readout beats a
# concentration readout, and it gets three more seeds before it is believed.
#
#   bash experiments/027-betaxanthin-metabolic/scripts/freeze_and_submit.sh \
#       experiments/027-betaxanthin-metabolic/scripts/gh_wave9_ibprec_e200_s2700.slurm
#
#SBATCH -p main
#SBATCH --mem=12g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --job-name=027-w9-ibprec-scale0-s2700
#SBATCH --time=10:00:00
#SBATCH --output=/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/kinetics-equilibrator-datasets/experiments/027-betaxanthin-metabolic/slurm/output/%x_%j.out
#SBATCH --error=/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/kinetics-equilibrator-datasets/experiments/027-betaxanthin-metabolic/slurm/output/%x_%j.out

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/kinetics-equilibrator-datasets}"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate torchcell
set -a; . "$PROJECT_ROOT/.env"; set +a
export EXPERIMENT_ROOT="$PROJECT_ROOT/experiments"
cd "$PROJECT_ROOT" || exit 1
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS=ignore
export WANDB_DIR=/scratch/projects/torchcell-scratch/experiments/027-betaxanthin-metabolic/wandb
export WANDB_PROJECT=torchcell_027_bxfx
export NUM_WORKERS=2
export COLLAPSE_PATIENCE_EPOCHS=200
export TC_TEST_PRED_DIR=$DATA_ROOT/test-predictions-027-wave7
mkdir -p "$WANDB_DIR"
echo "host $(hostname) gpu $CUDA_VISIBLE_DEVICES; DATA_ROOT=$DATA_ROOT; EXPERIMENT_ROOT=$EXPERIMENT_ROOT"
EPOCHS="${WAVE7_EPOCHS:-200}"

python experiments/027-betaxanthin-metabolic/scripts/train_bx.py \
    --arm flux_projection_anchored --kcat-tier predicted --thermo-source verified \
    --constraint-warmup-epochs 20 --constraint-scale 0.0 \
    --seed 2700 --epochs "$EPOCHS" --crps --graph-reg \
    --head isobutanol --ib-readout flux_precursor_concentration --ib-anchors one \
    --conc-space mixture --conc-upper-m 1.0 \
    --wandb-group gh_wave9_ib --tag _w9ibprec_scale0 \
    --out wave9_ibprec_scale0_e200_s2700.json
