#!/bin/bash
# experiments/028-gene-essentiality/scripts/gh_train_essentiality.slurm
# [[experiments.028-gene-essentiality.scripts.train_essentiality]]
#
# ONE LAUNCHER FOR EVERY 028 CELL, parameterized by environment. The 027 essentiality line
# carried sixteen near-identical per-cell files that drifted apart (the wave-9 launchers
# still requested 60g); this file replaces them. Every knob has a default that reproduces
# the wave-11 width-12 pooled cell, so an unset variable is a known cell, not a surprise.
#
#   ESS_ARM      pooled | flux_projection_anchored | flux_off | ...   (conf/arm_<name>.yaml)
#   ESS_HIDDEN   transformer width; must divide by ESS_HEADS AND by ESS_PERT_HEADS
#   ESS_HEADS    self-attention heads
#   ESS_LAYERS   transformer blocks
#   ESS_PREPROC  embedding-preprocessor depth (1 = a single projection; 2 = the 019 MLP)
#   ESS_SEED     weight-initialization seed; the split is pinned
#   ESS_EPOCHS   epoch ceiling
#   ESS_WD       AdamW weight decay (base.yaml: 1e-4)
#   ESS_LR       AdamW learning rate (base.yaml: 1e-3)
#   ESS_BATCH    batch size (base.yaml: 128)
#   ESS_SPLIT    a split file under results/splits/ (labels + pinned train/val/test); empty
#                keeps the wave-9 protocol (FCL labels, seeded 80/10/10 draw, test pinned)
#   ESS_TAG      the arm-level W&B group, e.g. _w12wd
#   ESS_OUT      the result file stem
#   ESS_EXTRA    anything else, verbatim (flux-arm flags, --apply-media dataset, ...)
#
#   ESS_ARM=pooled ESS_HIDDEN=12 ESS_HEADS=4 ESS_LAYERS=1 ESS_PREPROC=1 ESS_SEED=2700 \
#     ESS_TAG=_w12base ESS_OUT=w12_pool_h12l1p1_e80_s2700 \
#     bash experiments/028-gene-essentiality/scripts/freeze_and_submit.sh \
#          experiments/028-gene-essentiality/scripts/gh_train_essentiality.slurm
#
# MEMORY. 12g against a measured 5.4 GB peak resident set (wave-10 cells, cgroup
# memory.peak). The width-90 flux arm was measured at the same order; raise ESS_MEM for a
# larger flux cell rather than editing the default.
#
#SBATCH -p main
#SBATCH --mem=12g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --job-name=028-ess
#SBATCH --time=06:00:00
#SBATCH --output=/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/kinetics-equilibrator-datasets/experiments/028-gene-essentiality/slurm/output/%x_%j.out
#SBATCH --error=/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/kinetics-equilibrator-datasets/experiments/028-gene-essentiality/slurm/output/%x_%j.out

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
export WANDB_DIR=/scratch/projects/torchcell-scratch/experiments/028-gene-essentiality/wandb
export WANDB_PROJECT=torchcell_028_gene_essentiality
# DATA LOADER WORKERS, and why the default moved to 0 on 2026-09-16. Measured on a
# three-epoch CPU smoke of the width-12 split cell: 185 s wall with two spawned workers,
# 59 s with none, so about 42 s of every epoch was the workers re-importing the package
# (the multiprocessing context is spawn, and the loaders are rebuilt every epoch). A split
# cell holds 787 records; there is nothing to parallelize.
export NUM_WORKERS="${ESS_WORKERS:-0}"
export COLLAPSE_PATIENCE_EPOCHS=200
export TC_TEST_PRED_DIR=$DATA_ROOT/test-predictions-028
export TC_CHECKPOINT_DIR=$DATA_ROOT/checkpoints-028
mkdir -p "$WANDB_DIR" "$TC_TEST_PRED_DIR" "$TC_CHECKPOINT_DIR"
echo "host $(hostname) gpu $CUDA_VISIBLE_DEVICES; DATA_ROOT=$DATA_ROOT; EXPERIMENT_ROOT=$EXPERIMENT_ROOT"
echo "snapshot head: $(cat SNAPSHOT_GIT_HEAD 2>/dev/null || echo unfrozen)"

ARM="${ESS_ARM:-pooled}"
HIDDEN="${ESS_HIDDEN:-12}"
HEADS="${ESS_HEADS:-4}"
LAYERS="${ESS_LAYERS:-1}"
PREPROC="${ESS_PREPROC:-1}"
SEED="${ESS_SEED:-2700}"
EPOCHS="${ESS_EPOCHS:-80}"
TAG="${ESS_TAG:-_w12base}"
OUT="${ESS_OUT:-w12_${ARM}_h${HIDDEN}l${LAYERS}p${PREPROC}_e${EPOCHS}_s${SEED}}"

ARGS=()
[ -n "${ESS_WD:-}" ] && ARGS+=(--weight-decay "$ESS_WD")
[ -n "${ESS_LR:-}" ] && ARGS+=(--lr "$ESS_LR")
[ -n "${ESS_BATCH:-}" ] && ARGS+=(--batch-size "$ESS_BATCH")
[ -n "${ESS_SPLIT:-}" ] && ARGS+=(--split-file "$ESS_SPLIT")
# Graph regularization is ON unless ESS_GRAPH_REG=0; the class weight is OFF unless
# ESS_POSW=1. Both were fixed in every essentiality cell before wave 14.
[ "${ESS_GRAPH_REG:-1}" = "1" ] && ARGS+=(--graph-reg)
[ "${ESS_POSW:-0}" = "1" ] && ARGS+=(--pos-weight-balanced)

# shellcheck disable=SC2086
python experiments/028-gene-essentiality/scripts/train_essentiality.py \
    --arm "$ARM" --head essentiality \
    --seed "$SEED" --epochs "$EPOCHS" \
    --hidden "$HIDDEN" --attention-heads "$HEADS" --layers "$LAYERS" \
    --preprocessor-layers "$PREPROC" \
    "${ARGS[@]}" ${ESS_EXTRA:-} \
    --wandb-group gh_028_ess --tag "$TAG" \
    --out "${OUT}.json"
