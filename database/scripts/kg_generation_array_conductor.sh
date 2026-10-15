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
# of the order (a re-run of failed tasks into the existing directory, which keeps its
# order and fence, or one adapter for a measurement run into a new one).
# GEN_OVERRIDES carries hydra overrides to every task (a measurement knob);
# DEPENDENCY=afterany:<job> chains the arrays after a job.
#
# GEN_IMAGE_MODE (auto by default, or image, or install) decides how every task's
# container gets its torchcell, and the decision is made HERE, once, before any array
# is submitted (torchcell.knowledge_graphs.generation_image): `image` builds the tag
# when it is absent and refuses if it cannot; `install` leaves every task to run the
# three pip installs itself (255.9 s per task, lane C2 2026-10-15, against 15.8 s on
# the image path); `auto` takes the image when the tag already exists and the install
# path otherwise, and prints which. No task ever falls back on its own.
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
GEN_IMAGE_MODE="${GEN_IMAGE_MODE:-auto}"
JOB_NAME_PREFIX="${JOB_NAME_PREFIX:-gen_adapter}"
# DEPENDENCY=afterany:<job> chains every array of this submission after a job (a
# measurement run after the previous one); SBATCH_DEPENDENCY in the environment is
# NOT honored by slurm 24.05 here (jobs 4920/4921, 2026-10-15).
DEPENDENCY="${DEPENDENCY:-}"
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
GI="python -m torchcell.knowledge_graphs.generation_image"

echo "conductor: out $OUT_HOST; commit $BUILD_COMMIT; organisms $KG_ORGANISMS; include private $INCLUDE_PRIVATE; config $KG_CONFIG; overrides '$GEN_OVERRIDES'; dry run $DRY_RUN; indices '${INDICES:-all}'"

# The image decision, before any submission. `ensure` builds the tag when it is absent
# (about 4 min cold, free on the layer cache for the same commit, so a re-run of failed
# tasks pays no image time) and the tasks then start straight into the generator.
GEN_IMAGE=""
case "$GEN_IMAGE_MODE" in
    install) echo "image mode install: every task installs torchcell at $BUILD_COMMIT itself" ;;
    image|auto)
        RESOLVED=$($GI resolve --build-commit "$BUILD_COMMIT" --base-image "$IMAGE" --mode "$GEN_IMAGE_MODE" | grep -E '^RESOLVED_MODE=' | cut -d= -f2)
        [ -n "$RESOLVED" ] || { echo "$GI resolve printed no RESOLVED_MODE"; exit 1; }
        echo "image mode $GEN_IMAGE_MODE resolved to $RESOLVED"
        if [ "$RESOLVED" = image ]; then
            if [ "$DRY_RUN" = 1 ]; then
                echo "DRY-RUN $GI ensure --build-commit $BUILD_COMMIT --base-image $IMAGE --torchcell-src $TORCHCELL_SRC"
            else
                $GI ensure --build-commit "$BUILD_COMMIT" --base-image "$IMAGE" --torchcell-src "$TORCHCELL_SRC"
            fi
            GEN_IMAGE=$($GI tag --build-commit "$BUILD_COMMIT")
            echo "every task runs $GEN_IMAGE"
        fi
        ;;
    *) echo "GEN_IMAGE_MODE must be auto, image or install, got '$GEN_IMAGE_MODE'"; exit 1 ;;
esac
if [ -e "$OUT_HOST" ] && [ -z "$INDICES" ]; then
    echo "$OUT_HOST exists; a fresh array takes a new OUT_NAME (INDICES=<ids> re-runs into an existing one)"; exit 1
fi
if [ ! -e "$OUT_HOST" ]; then
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
    echo "reusing $OUT_HOST (order and fence recorded there)"
fi
$GA submit --out-name "$OUT_NAME" --out-dir "$OUT_HOST" --script "$ARRAY_SCRIPT" --tiers "$TIERS" \
    $PRIVATE_FLAG --organisms "$KG_ORGANISMS" ${INDICES:+--indices "$INDICES"} \
    --torchcell-src "$TORCHCELL_SRC" --build-commit "$BUILD_COMMIT" --dev-data-root "$DEV_DATA_ROOT" \
    --serve-root "$SERVE_ROOT" --build-root "$BUILD_ROOT" --kg-config "$KG_CONFIG" \
    ${GEN_OVERRIDES:+--gen-overrides "$GEN_OVERRIDES"} ${GEN_IMAGE:+--gen-image "$GEN_IMAGE"} \
    ${TCDB_TAGS:+--tcdb-tags "$TCDB_TAGS"} \
    --job-name-prefix "$JOB_NAME_PREFIX" ${DEPENDENCY:+--dependency "$DEPENDENCY"} $([ "$DRY_RUN" = 1 ] && echo --dry-run)
