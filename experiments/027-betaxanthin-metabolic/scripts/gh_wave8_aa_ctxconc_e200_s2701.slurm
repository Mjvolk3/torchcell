#!/bin/bash
# experiments/027-betaxanthin-metabolic/scripts/gh_wave8_aa_ctxconc_e200_s2701.slurm
# [[experiments.027-betaxanthin-metabolic.wave7-amino-acid-concentration]]
#
# WAVE 8 amino-acid cell: CONTEXT-ONLY concentration head, NO metabolic module, seed 2701.
#
# THE QUESTION (user, 2026-09-14): how do we know the metabolic module is ON for this
# target? The concentration readout is the module's only positive result on amino acids:
# +0.015 mean pinned Spearman over pooled (sd 0.009, n = 3, 200 epochs). But Finding 5 of
# [[experiments.027-betaxanthin-metabolic.wave8-sampling]] showed that readout predicts
# thermo_log_c = lo + (hi - lo) sigmoid(base + W context), a function of the genotype
# context alone: the flux never enters the prediction. The layer touches the concentration
# only through the thermodynamic penalty c_thermo (weight 1.0 x constraint scale 0.1 after a
# 20-epoch warmup), and at constraint scale 0 the gain was +0.020
# (wave7_aa_conc_nocstr_e200_s270{0,1,2}.json), so that coupling does not help.
#
# HYPOTHESIS (untested until this cell runs): the whole gain is the concentration
# PARAMETERIZATION, a per-metabolite-per-compartment sigmoid window between conc_lower_m
# and conc_upper_m, base + linear-from-context, then the volume-weighted mixture and log10,
# and the metabolic module contributes nothing beyond it.
#
# THE ABLATION. `--aa-readout context_concentration` on `--arm pooled` builds the SAME
# concentration head (log_c_base zeros over all 2,806 GEM metabolites, Linear(hidden, 2806)
# from the perturbed-gene pool, the same sigmoid window, the same metabolite index, the same
# compartment volume fractions, the same mixture, standardization and per-column CRPS
# log_sigma) directly on the pooled backbone WITHOUT constructing the flux layer: no
# stoichiometry, no kcat, no thermodynamics, no constraint term. From log c on it runs the
# flux readout's own code path (ConcentrationReadoutBase in train_bx.py, checked bit for bit
# in tests/torchcell/metabolism/test_context_concentration_readout.py).
#
# HOW TO READ IT. Pairs by seed with the wave-7 concentration cells at 200 epochs
# (wave7_aa_conc_e200_s2700/1/2: 0.0648 / 0.0663 / 0.0562 pinned mean Spearman) and with
# the wave-7 pooled cells (wave7_aa_pooled_e200_s2700/1/2: 0.0500 / 0.0425 / 0.0511). If this
# cell matches the concentration cells, the module is not on for this target; if it falls
# back to pooled, the module, through c_thermo at scale 0.1, is what carries the gain.
#
# Everything else is the wave-7 concentration recipe: 200 epochs, CRPS, graph
# regularizer, --head mulleder19, whole-cell mixture, window ceiling 1 M, the same
# centered 5-epoch moving-average epoch selection, the same pinned split. The flux-only
# flags (--kcat-tier, --thermo-source, --constraint-*) are dropped because there is no
# layer for them to configure.
#
#   bash experiments/027-betaxanthin-metabolic/scripts/freeze_and_submit.sh \
#       experiments/027-betaxanthin-metabolic/scripts/gh_wave8_aa_ctxconc_e200_s2701.slurm
#
#SBATCH -p main
#SBATCH --mem=60g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --job-name=027-w8-aa-ctxconc-s2701
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
    --arm pooled \
    --seed 2701 --epochs "$EPOCHS" --crps --graph-reg \
    --head mulleder19 --aa-readout context_concentration --conc-space mixture --conc-upper-m 1.0 \
    --wandb-group gh_wave8_aa --tag _w8aa_ctxconc \
    --out wave8_aa_ctxconc_e200_s2701.json
