#!/bin/bash
# experiments/027-betaxanthin-metabolic/scripts/gh_wave12_ess_grid.slurm
# [[experiments.027-betaxanthin-metabolic.wave12-essentiality-grid]]
#
# WAVE 12: two isolations on gene essentiality, one per objective.
#
#   A. THE BACKBONE (objective 1, best AUROC). Wave 11 moved width, depth and the
#      preprocessor depth together, so it cannot say which one bought the stability: the
#      1-layer, preprocessor-1 cells hold a median epoch-to-epoch validation step of 0.005
#      to 0.008 while every 2-layer, preprocessor-2 cell sits at 0.03 to 0.07, and the
#      unstable cells are the ones that collapse. Phase A crosses the factors, then moves
#      the levers no essentiality cell has moved: graph regularization off, the learning
#      rate, and the class weight.
#   B. THE MODULE (objective 2, can the flux layer help, or at least not hurt). Every
#      wave-9 flux cell ran the penalty configuration that the betaxanthin dose-response
#      showed collapses the layer's input (constraint scale 0.1), on the unstable width-90
#      backbone. Phase B puts the module on the stable backbone, with and without the
#      penalties, forced (biomass readout) and optional (gated readout), with the
#      stoichiometry-only arm and the shuffled-kcat control beside it.
#
# PARAMETERIZED BY ENVIRONMENT. Required: ESS_TAG, ESS_OUT. Everything else defaults to
# the phase-A/B reference cell (pooled, width 24, 8 heads, 1 layer, preprocessor 1, seed
# 2700, 80 epochs, graph reg on, lr from base.yaml, penalties off).
#
#   ESS_ARM=flux_projection_anchored ESS_READOUT=gated ESS_CSCALE=0.0 ESS_SEED=2701 \
#   ESS_TAG=_w12_gated_nopen ESS_OUT=wave12_ess_gated_nopen_h24l1p1_e80_s2701.json \
#       bash experiments/027-betaxanthin-metabolic/scripts/freeze_and_submit.sh \
#       experiments/027-betaxanthin-metabolic/scripts/gh_wave12_ess_grid.slurm
#
# Flux arms carry the wave-9 essentiality flux configuration (predicted kcat tier, the
# union-external thermodynamic source, 20 warmup epochs, the dataset medium, the mixture
# concentration space) with ONLY the constraint scale varied, so a phase-B cell differs
# from wave 9 by the backbone, the scale, the readout and, for the control, the kcat order.
#
# MEMORY. 12g against a measured peak of 5.4 GB on the wave-10 cells; the flux arms at
# width 90 fit the same request in wave 9.
#
#SBATCH -p main
#SBATCH --mem=12g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --job-name=027-w12-ess
#SBATCH --time=06:00:00
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

TAG="${ESS_TAG:?set ESS_TAG, the W&B tag and group suffix, e.g. _w12_a1}"
OUT="${ESS_OUT:?set ESS_OUT, the result file name, e.g. wave12_ess_pool_h90l1p1_e80_s2700.json}"
ARM="${ESS_ARM:-pooled}"
HIDDEN="${ESS_HIDDEN:-24}"
HEADS="${ESS_HEADS:-8}"
LAYERS="${ESS_LAYERS:-1}"
PREPROC="${ESS_PREPROC:-1}"
SEED="${ESS_SEED:-2700}"
EPOCHS="${ESS_EPOCHS:-80}"
READOUT="${ESS_READOUT:-biomass}"
CSCALE="${ESS_CSCALE:-0.0}"
GRAPH_REG="${ESS_GRAPH_REG:-1}"
POSW="${ESS_POSW:-0}"
LR="${ESS_LR:-}"
WD="${ESS_WD:-}"
SHUFFLE="${ESS_SHUFFLE:-0}"

EXTRA=()
[ "$GRAPH_REG" = "1" ] && EXTRA+=(--graph-reg)
[ "$POSW" = "1" ] && EXTRA+=(--pos-weight-balanced)
[ -n "$LR" ] && EXTRA+=(--lr "$LR")
[ -n "$WD" ] && EXTRA+=(--weight-decay "$WD")
if [ "$ARM" != "pooled" ]; then
    EXTRA+=(--kcat-tier predicted --thermo-source union_external
            --constraint-warmup-epochs 20 --constraint-scale "$CSCALE"
            --apply-media dataset --conc-space mixture --conc-upper-m 1.0
            --ess-readout "$READOUT")
    [ "$SHUFFLE" = "1" ] && EXTRA+=(--flux-shuffle-kcat "$SEED")
fi

echo "host $(hostname) gpu $CUDA_VISIBLE_DEVICES; arm $ARM hidden $HIDDEN heads $HEADS layers $LAYERS preproc $PREPROC seed $SEED readout $READOUT cscale $CSCALE graph_reg $GRAPH_REG posw $POSW lr '$LR' wd '$WD' shuffle $SHUFFLE"
echo "extra args: ${EXTRA[*]}"

python experiments/027-betaxanthin-metabolic/scripts/train_bx.py \
    --arm "$ARM" \
    --seed "$SEED" --epochs "$EPOCHS" \
    --head essentiality \
    --hidden "$HIDDEN" --attention-heads "$HEADS" --layers "$LAYERS" \
    --preprocessor-layers "$PREPROC" \
    --wandb-group "gh_wave12_ess${TAG}" --tag "$TAG" \
    "${EXTRA[@]}" \
    --out "$OUT"
