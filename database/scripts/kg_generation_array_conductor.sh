#!/bin/bash
# database/scripts/kg_generation_array_conductor.sh
# [[database.scripts.kg_generation_array_conductor]]
#
# Start a per-adapter generation: create $SERVE_ROOT/biocypher-out/$OUT_NAME (through a
# container, because biocypher-out is owned by uid 7474), record the build order and the
# dev-tree fence there, and submit one slurm array per resource tier of
# database/slurm/scripts/gilahyper_generate_adapter_array-slurm_docker.slurm. The last
# line printed is `JOB_IDS=<a>,<b>,<c>`, which the live rebuild chains its follow-up on.
#
#   OUT_NAME=<ts> TORCHCELL_SRC=<checkout> BUILD_COMMIT=<sha> KG_ORGANISMS=<set> \
#       INCLUDE_PRIVATE=1 bash database/scripts/kg_generation_array_conductor.sh
#
# DRY_RUN=1 prints every command (the mkdir, the fence, the sbatch lines) and submits
# nothing; INDICES=<slurm array spelling> restricts the submission to those positions
# of the order (a re-run of failed tasks, or one adapter for a measurement run), in
# which case the output directory must already exist and keeps its order and fence.
# GEN_OVERRIDES carries hydra overrides to every task (a measurement knob).
set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate torchcell

: "${OUT_NAME:?set OUT_NAME=<directory name under biocypher-out>}"
TORCHCELL_SRC="${TORCHCELL_SRC:-/home/michaelvolk/Documents/projects/torchcell}"
DEV_DATA_ROOT="${DEV_DATA_ROOT:-/scratch/projects/torchcell-scratch}"
BUILD_ROOT="${BUILD_ROOT:-/scratch/projects/torchcell}"
SERVE_ROOT="${SERVE_ROOT:-/db/database}"
IMAGE="${IMAGE:-michaelvolk/tc-neo4j:5.26.28-browser.1}"
INCLUDE_PRIVATE="${INCLUDE_PRIVATE:-1}"
KG_ORGANISMS="${KG_ORGANISMS:-scerevisiae,ecoli,pputida}"
KG_CONFIG="${KG_CONFIG:-kg_uncapped}"
GEN_OVERRIDES="${GEN_OVERRIDES:-}"
TIERS="${TIERS:-$TORCHCELL_SRC/torchcell/knowledge_graphs/conf/generation_tiers.yaml}"
ARRAY_SCRIPT="${ARRAY_SCRIPT:-$TORCHCELL_SRC/database/slurm/scripts/gilahyper_generate_adapter_array-slurm_docker.slurm}"
DRY_RUN="${DRY_RUN:-0}"
INDICES="${INDICES:-}"
JOB_NAME_PREFIX="${JOB_NAME_PREFIX:-gen_adapter}"
BUILD_COMMIT="${BUILD_COMMIT:-$(git -C "$TORCHCELL_SRC" rev-parse origin/main)}"
case "$INCLUDE_PRIVATE" in
    1) PRIVATE_FLAG="--include-private" ;;
    0) PRIVATE_FLAG="" ;;
    *) echo "INCLUDE_PRIVATE must be 0 or 1, got '$INCLUDE_PRIVATE'"; exit 1 ;;
esac
OUT_HOST="$SERVE_ROOT/biocypher-out/$OUT_NAME"
cd "$TORCHCELL_SRC"
export PYTHONPATH="$TORCHCELL_SRC"
GA="python -m torchcell.knowledge_graphs.generation_array"

echo "conductor: out $OUT_HOST; commit $BUILD_COMMIT; organisms $KG_ORGANISMS; include private $INCLUDE_PRIVATE; config $KG_CONFIG; overrides '$GEN_OVERRIDES'; dry run $DRY_RUN; indices '${INDICES:-all}'"
if [ -z "$INDICES" ]; then
    if [ -e "$OUT_HOST" ]; then echo "$OUT_HOST exists; a fresh array takes a new OUT_NAME"; exit 1; fi
    MKDIR="docker run --rm --entrypoint bash -v $SERVE_ROOT/biocypher-out:/out $IMAGE -c 'mkdir -p /out/$OUT_NAME/adapters && chown -R $(id -u):$(id -g) /out/$OUT_NAME && chmod -R 755 /out/$OUT_NAME'"
    if [ "$DRY_RUN" = 1 ]; then
        echo "DRY-RUN $MKDIR"
        echo "DRY-RUN $GA order $PRIVATE_FLAG --organisms $KG_ORGANISMS --write $OUT_HOST/adapter_order.json"
        echo "DRY-RUN $GA fence --dev-data-root $DEV_DATA_ROOT $PRIVATE_FLAG --organisms $KG_ORGANISMS > $OUT_HOST/dev_fence.txt"
    else
        eval "$MKDIR"
        $GA order $PRIVATE_FLAG --organisms "$KG_ORGANISMS" --write "$OUT_HOST/adapter_order.json" > "$OUT_HOST/adapter_order.txt"
        $GA fence --dev-data-root "$DEV_DATA_ROOT" $PRIVATE_FLAG --organisms "$KG_ORGANISMS" > "$OUT_HOST/dev_fence.txt"
        echo "order: $(wc -l < "$OUT_HOST/adapter_order.txt") adapters; fence: $(wc -l < "$OUT_HOST/dev_fence.txt") build manifests"
    fi
else
    for f in "$OUT_HOST/adapter_order.json" "$OUT_HOST/dev_fence.txt"; do
        [ -f "$f" ] || { echo "$f is absent; a re-run needs the directory the first submission created"; exit 1; }
    done
fi
$GA submit --out-name "$OUT_NAME" --out-dir "$OUT_HOST" --script "$ARRAY_SCRIPT" --tiers "$TIERS" \
    $PRIVATE_FLAG --organisms "$KG_ORGANISMS" ${INDICES:+--indices "$INDICES"} \
    --torchcell-src "$TORCHCELL_SRC" --build-commit "$BUILD_COMMIT" --dev-data-root "$DEV_DATA_ROOT" \
    --serve-root "$SERVE_ROOT" --build-root "$BUILD_ROOT" --kg-config "$KG_CONFIG" \
    ${GEN_OVERRIDES:+--gen-overrides "$GEN_OVERRIDES"} ${TCDB_TAGS:+--tcdb-tags "$TCDB_TAGS"} \
    --job-name-prefix "$JOB_NAME_PREFIX" $([ "$DRY_RUN" = 1 ] && echo --dry-run)
