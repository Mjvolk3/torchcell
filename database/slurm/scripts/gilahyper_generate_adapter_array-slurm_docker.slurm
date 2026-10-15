#!/bin/bash
#SBATCH --job-name=gen_adapter
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/projects/torchcell/database/slurm/output/%A_%a_%x.out
#SBATCH --error=/scratch/projects/torchcell/database/slurm/output/%A_%a_%x.out
#SBATCH --no-requeue

# ONE adapter's CSV generation, one array task per adapter: the per-adapter generation
# of the live rebuild (GENERATION_MODE=array in gilahyper_live_rebuild-slurm_docker.slurm)
# and of a measurement run (the Hoepfner study). Task N generates position N of the
# build order (torchcell.knowledge_graphs.generation_array.adapter_order: the public
# map, then the private map, narrowed to KG_ORGANISMS) in its own container
# tc-neo4j-gen-N, into $SERVE_ROOT/biocypher-out/$OUT_NAME/adapters/NNN_<Adapter>/,
# with its own telemetry, and writes the DONE marker last.
#
# RECOVERY. A task that fails leaves no DONE marker; every other directory is complete
# and untouched. After the fix, re-run exactly the missing tasks:
#
#   PYTHONPATH=$TORCHCELL_SRC python -m torchcell.knowledge_graphs.generation_array missing \
#       --out-dir $SERVE_ROOT/biocypher-out/$OUT_NAME --out-name $OUT_NAME \
#       --torchcell-src $TORCHCELL_SRC --build-commit $BUILD_COMMIT [--include-private] --organisms <set>
#
# prints the indices without a marker and, per tier, the exact
# `... sbatch --array=<ids>%N ... gilahyper_generate_adapter_array-slurm_docker.slurm`
# line (the environment assignments first; multi-member values never go inside
# --export's list, which sbatch splits on commas). Then the merge:
#
#   python -m torchcell.knowledge_graphs.generation_array merge --out-dir <out> --workers 16
#
# refuses while any directory lacks its marker and names them; the live rebuild's
# RESUME_FROM=array does the same re-run and merge in one job.
#
# SUBMISSION. Do not submit this script by hand with a bare --array: the conductor
# (database/scripts/kg_generation_array_conductor.sh, or `generation_array submit`)
# creates the output directory, writes adapter_order.json and dev_fence.txt there, and
# submits one array per resource tier (torchcell/knowledge_graphs/conf/generation_tiers.yaml),
# because an array carries one resource spec. The tier's values arrive as GEN_CPUS and
# GEN_MEM_GB and are the container's --cpus and --memory: docker runs outside the slurm
# cgroup, so the slurm --mem is a reservation and the docker limit is the real cap; the
# conductor sets both from the same tier (MaxMemPerCPU=4100 may round the allocation's
# CPU count up; the container keeps the tier's). The header above is the standard tier;
# the conductor overrides it per tier on the command line.
#
# FENCE. The conductor digests every mapped dev store's build manifest once
# (dev_fence.txt); this task re-digests before it generates and refuses on a
# difference, so a store rebuilt under the array is caught per task, not after 12 h.
#
# CONTAINER. As in the live rebuild: the dev tree's data/torchcell, ecoli, pputida and
# torchcell-genomes mounted read-only, the build tree's biocypher config, torchcell
# pip-installed from GitHub at BUILD_COMMIT. `scancel` kills this script, not the
# container (section 0 of the torchcell-db-build skill): remove tc-neo4j-gen-N by
# hand after a cancel, and never leave one running without a reservation behind it.
#
# Variables (set by the conductor; all required unless a default is shown):
#   OUT_NAME        the directory name under $SERVE_ROOT/biocypher-out
#   GEN_CPUS        container --cpus and the adapter's worker count
#   GEN_MEM_GB      container --memory in GiB
#   GEN_TIER        the tier's name (logging only)
#   TORCHCELL_SRC   the checkout at BUILD_COMMIT (default: the primary checkout)
#   BUILD_COMMIT    the commit the container installs (default: origin/main of TORCHCELL_SRC)
#   KG_ORGANISMS    the organism set (default scerevisiae,ecoli,pputida)
#   INCLUDE_PRIVATE 1 (default) or 0
#   KG_CONFIG       the hydra config (default kg_uncapped)
#   GEN_OVERRIDES   extra hydra overrides, space-separated (a measurement knob such as
#                   `adapters.single_pass_chunk_budget_mb=256`); default none
#   DEV_DATA_ROOT, BUILD_ROOT, SERVE_ROOT, IMAGE as in the live rebuild

source ~/miniconda3/etc/profile.d/conda.sh
conda activate torchcell
set -euo pipefail

INDEX="${SLURM_ARRAY_TASK_ID:?not an array job}"
: "${OUT_NAME:?set OUT_NAME (the conductor does)}"
: "${GEN_CPUS:?set GEN_CPUS (the conductor does, from the tier)}"
: "${GEN_MEM_GB:?set GEN_MEM_GB (the conductor does, from the tier)}"
GEN_TIER="${GEN_TIER:-unnamed}"
KG_CONFIG="${KG_CONFIG:-kg_uncapped}"
TORCHCELL_SRC="${TORCHCELL_SRC:-/home/michaelvolk/Documents/projects/torchcell}"
DEV_DATA_ROOT="${DEV_DATA_ROOT:-/scratch/projects/torchcell-scratch}"
BUILD_ROOT="${BUILD_ROOT:-/scratch/projects/torchcell}"
SERVE_ROOT="${SERVE_ROOT:-/db/database}"
IMAGE="${IMAGE:-michaelvolk/tc-neo4j:5.26.28-browser.1}"
INCLUDE_PRIVATE="${INCLUDE_PRIVATE:-1}"
case "$INCLUDE_PRIVATE" in
    1) PRIVATE_FLAG="--include-private" ;;
    0) PRIVATE_FLAG="" ;;
    *) echo "INCLUDE_PRIVATE must be 0 or 1, got '$INCLUDE_PRIVATE'"; exit 1 ;;
esac
KG_ORGANISMS="${KG_ORGANISMS:-scerevisiae,ecoli,pputida}"
GEN_OVERRIDES="${GEN_OVERRIDES:-}"
BUILD_COMMIT="${BUILD_COMMIT:-$(git -C "$TORCHCELL_SRC" rev-parse origin/main)}"
PIP_REF="git+https://github.com/Mjvolk3/torchcell.git@$BUILD_COMMIT"
OUTDIR="$BUILD_ROOT/database/slurm/output"
OUT_HOST="$SERVE_ROOT/biocypher-out/$OUT_NAME"
OUT_CONT="/var/lib/neo4j/biocypher-out/$OUT_NAME"
GEN_CONTAINER="tc-neo4j-gen-$INDEX"
JOB="${SLURM_ARRAY_JOB_ID:-nojob}_$INDEX"
cd "$TORCHCELL_SRC"
export PYTHONPATH="$TORCHCELL_SRC"

echo "Running on $(hostname) at $(date); array job ${SLURM_ARRAY_JOB_ID:-?} task $INDEX (tier $GEN_TIER: $GEN_CPUS cpus, ${GEN_MEM_GB}G)"
echo "config $KG_CONFIG; commit $BUILD_COMMIT; dev tree $DEV_DATA_ROOT; out $OUT_HOST; include private $INCLUDE_PRIVATE; organisms $KG_ORGANISMS; overrides '$GEN_OVERRIDES'"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-?} SLURM_MEM_PER_NODE=${SLURM_MEM_PER_NODE:-?}"

echo "---------------- 0 PREFLIGHT (task $INDEX)"
for f in "$OUT_HOST/adapter_order.json" "$OUT_HOST/dev_fence.txt"; do
    [ -f "$f" ] || { echo "$f is absent: submit through the conductor, not by hand"; exit 1; }
done
DIRNAME=$(python - "$OUT_HOST/adapter_order.json" "$INDEX" <<'EOF'
import sys
from pathlib import Path
from torchcell.knowledge_graphs.generation_array import read_order, slot_by_index
print(slot_by_index(read_order(Path(sys.argv[1])), int(sys.argv[2])).dirname)
EOF
)
echo "adapter directory: $DIRNAME"
if [ -f "$OUT_HOST/adapters/$DIRNAME/DONE" ]; then
    echo "$OUT_HOST/adapters/$DIRNAME already carries DONE; nothing to do"; exit 0
fi
if docker inspect "$GEN_CONTAINER" >/dev/null 2>&1; then
    echo "$GEN_CONTAINER already exists (a cancelled task's container outlives its job); docker rm -f it first"; exit 1
fi
# The fingerprint repo must carry the same code the container installs.
if ! git -C "$TORCHCELL_SRC" diff --quiet "$BUILD_COMMIT" HEAD -- torchcell biocypher; then
    echo "$TORCHCELL_SRC HEAD differs from $BUILD_COMMIT under torchcell/ or biocypher/"; exit 1
fi
FENCE_NOW="$OUTDIR/${JOB}_dev_fence_now.txt"
python -m torchcell.knowledge_graphs.generation_array fence --dev-data-root "$DEV_DATA_ROOT" \
    $PRIVATE_FLAG --organisms "$KG_ORGANISMS" > "$FENCE_NOW"
if diff "$OUT_HOST/dev_fence.txt" "$FENCE_NOW"; then
    echo "dev fence intact: $(wc -l < "$FENCE_NOW") build manifests unchanged since the conductor"
else
    echo "a dev store changed since the conductor digested the tree (diff above); this task refuses"
    exit 1
fi
# A previous attempt may have left a partial directory (no DONE): keep it aside, never
# write into it, so the marker protocol stays "DONE means everything is there".
if [ -d "$OUT_HOST/adapters/$DIRNAME" ]; then
    ASIDE="$OUT_HOST/adapters/$DIRNAME.incomplete.$(date +%Y-%m-%d_%H-%M-%S)"
    docker run --rm --entrypoint bash -v "$OUT_HOST/adapters":/a "$IMAGE" -c \
        "mv /a/$DIRNAME /a/$(basename "$ASIDE")"
    echo "previous incomplete attempt kept at $ASIDE"
fi

echo "---------------- 1 GENERATE $DIRNAME in $GEN_CONTAINER $(date)"
GEN_LOG="$OUTDIR/${JOB}_generate.log"
docker run --rm --entrypoint bash --name "$GEN_CONTAINER" \
    --cpus="$GEN_CPUS" -e SLURM_CPUS_PER_TASK="$GEN_CPUS" --memory="${GEN_MEM_GB}g" \
    -e NEO4J_ACCEPT_LICENSE_AGREEMENT=yes \
    -e BUILD_COMMIT="$BUILD_COMMIT" \
    -e SLURM_ARRAY_JOB_ID="${SLURM_ARRAY_JOB_ID:-}" -e SLURM_ARRAY_TASK_ID="$INDEX" \
    -e TCDB_TAGS="${TCDB_TAGS:-}" -e TCDB_JOB_TYPE="${TCDB_JOB_TYPE:-generate_adapter}" \
    -v "$SERVE_ROOT/biocypher-out":/var/lib/neo4j/biocypher-out \
    -v "$BUILD_ROOT/database/data":/var/lib/neo4j/data \
    -v "$DEV_DATA_ROOT/data/torchcell":/var/lib/neo4j/data/torchcell:ro \
    -v "$DEV_DATA_ROOT/data/ecoli":/var/lib/neo4j/data/ecoli:ro \
    -v "$DEV_DATA_ROOT/data/pputida":/var/lib/neo4j/data/pputida:ro \
    -v "$DEV_DATA_ROOT/torchcell-genomes":/var/lib/neo4j/torchcell-genomes:ro \
    -v "$BUILD_ROOT/database/.env":/.env \
    -v "$BUILD_ROOT/database/biocypher":/var/lib/neo4j/biocypher \
    "$IMAGE" -c "set -euo pipefail
        cd /var/lib/neo4j
        python -m pip install --no-cache-dir --force-reinstall --no-deps '$PIP_REF' >/dev/null
        python -m pip install --no-cache-dir biocypher==0.15.2 >/dev/null
        python -m pip install --no-cache-dir git+https://github.com/oxpig/CaLM@main >/dev/null
        python -c \"import importlib.resources as r, sys; p = r.files('torchcell').joinpath('knowledge_graphs/conf/${KG_CONFIG}.yaml'); sys.exit(0 if p.is_file() else 1)\" \
            || { echo '${KG_CONFIG}.yaml is not in the installed torchcell'; exit 1; }
        set -a; source /.env; set +a
        wandb login \$WANDB_API_KEY >/dev/null
        mkdir -p $OUT_CONT/adapters/$DIRNAME
        python -m torchcell.knowledge_graphs.generate_adapter --config-name $KG_CONFIG \
            --adapter-index $INDEX --out-dir $OUT_CONT/adapters/$DIRNAME \
            $PRIVATE_FLAG --organisms $KG_ORGANISMS $GEN_OVERRIDES
        chown -R 7474:7474 $OUT_CONT/adapters/$DIRNAME && chmod -R a+rX $OUT_CONT/adapters/$DIRNAME
        echo GENERATION_DIR=$OUT_CONT/adapters/$DIRNAME" 2>&1 | tee "$GEN_LOG"
[ -f "$OUT_HOST/adapters/$DIRNAME/DONE" ] || { echo "no DONE marker in $OUT_HOST/adapters/$DIRNAME"; exit 1; }
echo "DONE task $INDEX $DIRNAME at $(date): $(tr -d '\n' < "$OUT_HOST/adapters/$DIRNAME/DONE" | cut -c1-400)"
