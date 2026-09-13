#!/bin/bash
# experiments/027-betaxanthin-metabolic/scripts/gh_wave8_aa_concver_e200_s2700.slurm
# [[experiments.027-betaxanthin-metabolic.wave8]]
#
# WAVE 8 BLOCK 1: the CONCENTRATION readout on the VERIFIED reaction-energy tables, seed 2700.
#
# THE ARM UNDER TEST. `flux_projection_anchored` (mass balance exact by construction,
# anchored thermodynamics on the union_external table, the predicted-kcat capacity bound
# through the abundance head, the protein-pool budget and the dissipation limit) at
# constraint scale 0.1 over a 20-epoch warmup, with `--aa-readout flux_concentration`.
#
# WHAT THE READOUT IS. The prediction for amino acid k is log10 of the layer's OWN learned
# WHOLE-CELL concentration of that amino acid, sum_n f_n [A]_n over every intracellular
# compartment the GEM carries it in, weighted by fractional volume, with no parameters on
# the mean. Mulleder measures a whole-cell concentration while log_c is per compartment,
# and arginine, lysine and histidine are largely vacuolar in yeast, so a cytosol-only
# readout would predict the wrong pool for them (Thermo-Flux Box 2 states the relation;
# the fractional volumes and their provenance are in train_bx.compartment_volume_fractions
# and are written into every result file).
#
# The same tensor sets every reaction energy through RT S^T ln c, so the amino-acid
# gradient and the second law read one quantity. Wave 6's readout instead mapped the
# metabolite's TURNOVER through a free per-column affine and measured -0.039 mean
# per-feature Spearman against pooled at three seeds.
#
# WHY THE WINDOW IS WIDENED. The anchored concentration is squashed into
# [conc_lower_m, conc_upper_m], shipped as 1e-7 to 1e-2 M. Mulleder's measured intracellular
# amino acids run to 9.5e-2 M and 10.6 percent of all 88,882 measurements are above 1e-2 M,
# with glutamate and glutamine above it in 99.8 percent of strains, so under the shipped
# ceiling the supervised value would be unreachable and the sigmoid would saturate silently
# (mulleder_concentration_window.py, results/mulleder_concentration_window.json). The
# ceiling is raised to 1 M, a decade above the largest measurement, for EVERY wave-7 flux
# cell including the turnover arm, so the turnover-versus-concentration contrast is on the
# readout alone and not on the window.
#
# 200 EPOCHS, not 80, and the reason is measured. On the 80-epoch cells the concentration
# head was STILL RISING at the last epoch (job 1746 seed 2700: validation Pearson max 0.131
# at epoch 79, validation loss minimum at 79, train data loss still falling 0.71 to 0.50;
# job 1747 seed 2701: max 0.135 at epoch 75), while the pooled control peaks at epoch 23 to
# 27 and then overfits. Selecting an epoch inside a budget one arm has not finished using
# biases the comparison against that arm, so every wave-7 200-epoch cell, control included,
# gets the same budget. The scoring rule is unchanged: the epoch is the argmax of the
# centered 5-epoch moving average of validation Pearson, never a maximum over epochs.
#
# WHAT CHANGES, AND IT IS ONLY THE THERMODYNAMIC TABLE. `--thermo-source verified` reads
# the curated tables under `thermo_equilibrator/verified/`, pinned by their manifest sha256:
# 3,801 of 4,131 reactions priced against union_external's 3,779, reaction values read from
# the verified REACTION table rather than re-summed over S, and the only route where the
# accession composition guard has run, so the glycogen and dolichol mispricings are absent.
# Every other field matches the wave-7 cell it is paired against
# (`wave7_aa_conc_e200_s<seed>.json`, same seed), so the contrast is the table alone.
#
# NOT ON THE WAVE-6 FROZEN TREE: freeze_and_submit.sh snapshots the code per job, so this
# cell carries the shipped verified kcat table (97 measured / 782 homolog / 2,831 predicted
# / 18 default), which every wave-7 result file records under flux_meta.kcat_tier_counts.
#
#   bash experiments/027-betaxanthin-metabolic/scripts/freeze_and_submit.sh \
#       experiments/027-betaxanthin-metabolic/scripts/gh_wave8_aa_concver_e200_s2700.slurm
#
#SBATCH -p main
#SBATCH --mem=60g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --job-name=027-w8-aa-concver-s2700
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
    --constraint-warmup-epochs 20 --constraint-scale 0.1 \
    --seed 2700 --epochs "$EPOCHS" --crps --graph-reg \
    --head mulleder19 --aa-readout flux_concentration --conc-space mixture --conc-upper-m 1.0 \
    --wandb-group gh_wave8_aa --tag _w8aa \
    --out wave8_aa_concver_e200_s2700.json
