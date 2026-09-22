#!/bin/bash
# experiments/028-gene-essentiality/scripts/ess_cell.sh
# [[experiments.028-gene-essentiality.scripts.train_essentiality]]
#
# ONE CELL, from its ESS_* environment to one train_essentiality.py invocation. This is the
# body that gh_train_essentiality.slurm ran inline until 2026-09-16; it is a file of its own
# so that the single-cell launcher and the pack launcher (gh_train_essentiality_pack.slurm,
# several cells on one GPU) run the same lines and cannot drift. The caller has already
# activated the environment, sourced .env, exported the WANDB/TC_* variables and cd'd to
# PROJECT_ROOT. The knobs are documented in gh_train_essentiality.slurm.
set -u

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

echo "cell $OUT: arm $ARM h$HIDDEN l$LAYERS p$PREPROC seed $SEED epochs $EPOCHS tag $TAG"
# shellcheck disable=SC2086
python experiments/028-gene-essentiality/scripts/train_essentiality.py \
    --arm "$ARM" --head "${ESS_HEAD:-essentiality}" \
    --seed "$SEED" --epochs "$EPOCHS" \
    --hidden "$HIDDEN" --attention-heads "$HEADS" --layers "$LAYERS" \
    --preprocessor-layers "$PREPROC" \
    "${ARGS[@]}" ${ESS_EXTRA:-} \
    --wandb-group gh_028_ess --tag "$TAG" \
    --out "${OUT}.json"
