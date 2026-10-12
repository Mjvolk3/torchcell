#!/bin/bash
# experiments/027-betaxanthin-metabolic/scripts/gh_wave9_aa_holdout_ctx_e200_s2702.slurm
# [[experiments.027-betaxanthin-metabolic.wave9-isobutanol]]
#
# WAVE 9 BLOCK 3: HELD-OUT AMINO-ACID COLUMNS, `context-only (`context_concentration`), the layer-free control`, seed 2702.
#
# THE QUESTION. Is the learned log c ONE physical state or 19 independent heads? A physical
# concentration is a single object linked by stoichiometry and thermodynamics, so supervision
# at 13 columns should reach the other 6 through S and the reaction energies. A bank of 2,806
# independent sigmoids cannot: rows the mixture never reads receive no gradient and stay at
# their initialization, which is asserted directly in
# tests/torchcell/metabolism/test_context_concentration_readout.py.
#
# WHAT IS HELD OUT: valine, tyrosine, glutamate, leucine, isoleucine, methionine. The first
# three are the multi-target anchors -- valine is one transaminase step from the isobutanol
# biosensor's pool, tyrosine is the betaxanthin cassette's substrate, and glutamate is the
# amino donor for both transaminations -- so these are exactly the rows a shared latent would
# have to carry.
#
# HOW THE MASK IS APPLIED. The LOSS and the epoch-selection metric are charged on the 13
# supervised columns only; the test scorer is unchanged and reports all 19 per column, with
# the two groups summarized in `aa_column_split`. Selecting the epoch on a held-out column's
# validation Pearson would be choosing the checkpoint by the very quantity the arm asks about.
#
# CONTROL ALREADY MEASURED: the UNSUPERVISED baseline on all 19 is -0.0041 and +0.0071, read
# off the turnover cells 1766 and 1772, whose anchored layer learns a concentration while
# being trained on turnover.
#
# PRE-REGISTERED READ: mean pinned-test Spearman over the 6 held-out columns, flux-coupled vs
# context-only vs unsupervised. SUCCESS/KILL: flux-coupled held-out above 0.02 AND above
# context-only by more than 0.010 means the latent is a coupled state, which a pooled backbone
# structurally cannot produce. Both near zero means the readout is 19 independent sigmoids;
# the "learned concentration" framing is retired and the budget moves elsewhere. If the
# context-only arm ALSO transfers, the transfer is coming from the shared backbone rather than
# from the layer, and it is reported as such.
#
#   bash experiments/027-betaxanthin-metabolic/scripts/freeze_and_submit.sh \
#       experiments/027-betaxanthin-metabolic/scripts/gh_wave9_aa_holdout_ctx_e200_s2702.slurm
#
#SBATCH -p main
#SBATCH --mem=12g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --job-name=027-w9-aahold-ctx-s2702
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
    --seed 2702 --epochs "$EPOCHS" --crps --graph-reg \
    --head mulleder19 --aa-readout context_concentration --conc-space mixture --conc-upper-m 1.0 \
    --aa-supervise-columns alanine,arginine,asparagine,aspartate,glutamine,glycine,histidine,lysine,phenylalanine,proline,serine,threonine,tryptophan \
    --wandb-group gh_wave9_aa --tag _w9aaholdout_ctx \
    --out wave9_aa_holdout_ctx_e200_s2702.json
