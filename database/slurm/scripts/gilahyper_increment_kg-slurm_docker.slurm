#!/bin/bash
#SBATCH --job-name=increment_kg
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/projects/torchcell/database/slurm/output/%j_%x.out
#SBATCH --error=/scratch/projects/torchcell/database/slurm/output/%j_%x.out
#SBATCH --no-requeue

# Add ONE dataset to the SERVED knowledge graph without rebuilding the others.
#
# Resources: small on purpose. The heavy steps run in docker containers whose memory is
# bounded by their own --memory limits (CSV generation) or by the serving container's
# limit (the import: HEAP_SIZE + --max-off-heap-memory), not by this allocation; the
# slurm request covers the host-side admission check, staging and verification. Keeping
# it small also lets the backfill scheduler place it beside the GPU sweeps.
#
#   sbatch --export=ALL,DATASET_CLASS=NadalRibellesPerturbSeq2025Dataset \
#       database/slurm/scripts/gilahyper_increment_kg-slurm_docker.slurm
#
# Stages (each one fails loudly; nothing is retried or worked around):
#   1 ADMIT    kg_manifest admit: the served datasets' schema closures, the graph schema
#              and the adapter code must be unchanged for everything already served, and
#              the new dataset's dev-tree LMDB must be fresh. A block means FULL REBUILD
#              (or, for adapter drift only, a reviewed ACK_ADAPTER_DRIFT reason).
#   2 STAGE    copy the dev-tree LMDB into the uid-7474 build tree (via a container).
#   3 REFRESH  directory_setup: the build tree's biocypher/ config copy (schema config
#              with the new node classes) and .env are refreshed from TORCHCELL_SRC.
#   4 GENERATE an ephemeral python container (no Neo4j server) runs the KG builder in
#              incremental mode for this dataset only; CSVs land in
#              $SERVE_ROOT/biocypher-out/<ts>, which the SERVING container mounts.
#   5 IMPORT   in the serving container: create the id uniqueness constraints (the
#              database is made writable for that step through the dynamic
#              server.databases.writable setting, then set back), STOP DATABASE,
#              neo4j-admin database import incremental as the neo4j user, START DATABASE.
#   6 VERIFY   the new Dataset node's ExperimentMemberOf count equals the CSV row count.
#   7 RECORD   kg_manifest record; CSVs archived to /bulk/biocypher-out/<ts>.
#
# Variables (all optional except DATASET_CLASS):
#   TORCHCELL_SRC      checkout providing the host-side code + .env (default: main)
#   TORCHCELL_PIP_REF  what the CSV-generating container installs
#                      (default git+https://github.com/Mjvolk3/torchcell.git@main;
#                      set TORCHCELL_PIP_REF=src to install TORCHCELL_SRC itself, for a
#                      dress rehearsal from a worktree before landing)
#   KG_MANIFEST        the served store's manifest (default $BUILD_ROOT/database/kg_manifest.json)
#   BUILD_ROOT         dataset build tree root (default /scratch/projects/torchcell)
#   SERVE_ROOT         serving root the container mounts (default /db/database)
#   NEO4J_CONTAINER    serving container (default tc-neo4j-readonly)
#   NEO4J_DATABASE     database to grow (default torchcell)
#   ACK_ADAPTER_DRIFT  written reason passed to --ack-adapter-drift (default: none)
#   ARCHIVE_ROOT       CSV archive (default /bulk/biocypher-out)
#   DATABASE_ENV_FILE  the container .env (untracked secrets; default: the primary
#                      checkout's database/database.env, which a worktree lacks)
#
# Recovery: a failed neo4j-admin incremental import can leave the store unusable
# (Neo4j's own warning). The CSV archive under /bulk/biocypher-out holds every full and
# incremental import ever applied, so the store is regenerable: re-run the full import
# call of the last full build, then each archived increment in order.

set -euo pipefail
echo "Running on $(hostname) at $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate torchcell

# Runs under sbatch (the SLURM_* values come from the allocation) or directly from a
# shell when the queue cannot place it, in which case the same small envelope applies.
SLURM_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-8}"
SLURM_MEM_PER_NODE="${SLURM_MEM_PER_NODE:-32768}"

: "${DATASET_CLASS:?set DATASET_CLASS=<registered dataset class name>}"
TORCHCELL_SRC="${TORCHCELL_SRC:-/home/michaelvolk/Documents/projects/torchcell}"
TORCHCELL_PIP_REF="${TORCHCELL_PIP_REF:-git+https://github.com/Mjvolk3/torchcell.git@main}"
BUILD_ROOT="${BUILD_ROOT:-/scratch/projects/torchcell}"
SERVE_ROOT="${SERVE_ROOT:-/db/database}"
KG_MANIFEST="${KG_MANIFEST:-$BUILD_ROOT/database/kg_manifest.json}"
NEO4J_CONTAINER="${NEO4J_CONTAINER:-tc-neo4j-readonly}"
NEO4J_DATABASE="${NEO4J_DATABASE:-torchcell}"
ARCHIVE_ROOT="${ARCHIVE_ROOT:-/bulk/biocypher-out}"
DATABASE_ENV_FILE="${DATABASE_ENV_FILE:-/home/michaelvolk/Documents/projects/torchcell/database/database.env}"
IMAGE=michaelvolk/tc-neo4j:5.26.28
CS='cypher-shell -u neo4j -p torchcell'

cd "$TORCHCELL_SRC"
export PYTHONPATH="$TORCHCELL_SRC"
DEV_DATA_ROOT=$(grep -E '^DATA_ROOT=' .env | cut -d= -f2- | tr -d '"')
JOB="${SLURM_JOB_ID:-nojob}"
REPORT="$BUILD_ROOT/database/slurm/output/${JOB}_admission_${DATASET_CLASS}.json"
echo "source $TORCHCELL_SRC ($(git rev-parse --short HEAD)); dev tree $DEV_DATA_ROOT"
echo "dataset $DATASET_CLASS -> $NEO4J_CONTAINER/$NEO4J_DATABASE; manifest $KG_MANIFEST"

echo "---------------- 0 PREFLIGHT"
[ -f "$KG_MANIFEST" ] || { echo "no manifest at $KG_MANIFEST (bootstrap it first)"; exit 1; }
docker inspect "$NEO4J_CONTAINER" >/dev/null || { echo "container $NEO4J_CONTAINER not found"; exit 1; }
docker inspect "$NEO4J_CONTAINER" --format '{{range .Mounts}}{{.Destination}}{{"\n"}}{{end}}' \
    | grep -qx /var/lib/neo4j/biocypher-out \
    || { echo "$NEO4J_CONTAINER does not mount /var/lib/neo4j/biocypher-out; relaunch it with -v $SERVE_ROOT/biocypher-out:/var/lib/neo4j/biocypher-out"; exit 1; }
docker exec "$NEO4J_CONTAINER" $CS -d system "SHOW DATABASE $NEO4J_DATABASE YIELD name, currentStatus, access;"
SLUG=$(python -c "
from torchcell.database.build_dataset_lmdb import resolve_dataset_class, dataset_default_root
import os.path as osp; print(osp.basename(dataset_default_root(resolve_dataset_class('$DATASET_CLASS'))))")
echo "slug $SLUG"

echo "---------------- 1 ADMIT"
ACK_ARGS=()
if [ -n "${ACK_ADAPTER_DRIFT:-}" ]; then ACK_ARGS=(--ack-adapter-drift "$ACK_ADAPTER_DRIFT"); fi
python -m torchcell.knowledge_graphs.kg_manifest --manifest "$KG_MANIFEST" --repo "$TORCHCELL_SRC" \
    admit --dataset "$DATASET_CLASS" --data-root "$DEV_DATA_ROOT" --report "$REPORT" "${ACK_ARGS[@]}"

echo "---------------- 2 STAGE dev LMDB -> build tree (as uid 7474)"
TS=$(date +%Y-%m-%d_%H-%M-%S)
# --entrypoint bash bypasses the image entrypoint that drops to the neo4j user, so the
# uid is set explicitly: the build tree is uid-7474 territory.
docker run --rm --entrypoint bash --user 7474:7474 \
    -v "$BUILD_ROOT/database/data":/var/lib/neo4j/data \
    -v "$DEV_DATA_ROOT/data/torchcell/$SLUG":/mnt/src:ro \
    "$IMAGE" -c "set -euo pipefail
        BT=/var/lib/neo4j/data/torchcell/$SLUG; mkdir -p \$BT
        for d in processed preprocess; do
            if [ -e \$BT/\$d ]; then mv \$BT/\$d \$BT/\$d.superseded.$TS; echo \"moved aside \$BT/\$d\"; fi
            cp -a /mnt/src/\$d \$BT/\$d
        done
        ls -la \$BT \$BT/processed"

echo "---------------- 3 REFRESH build-tree biocypher config + env from $TORCHCELL_SRC"
DATA_ROOT="$BUILD_ROOT" python -m torchcell.database.directory_setup --env-file "$DATABASE_ENV_FILE"
cmp "$TORCHCELL_SRC/biocypher/config/torchcell_schema_config.yaml" \
    "$BUILD_ROOT/database/biocypher/config/torchcell_schema_config.yaml"

echo "---------------- 4 GENERATE CSVs (ephemeral container, incremental mode)"
# biocypher-out is written by uid 7474 (the generating container) and READ by this host
# user (reference analysis, counts), so it is 7474-owned and world-readable.
docker run --rm --user 0 -v "$SERVE_ROOT":/serve "$IMAGE" bash -c \
    'mkdir -p /serve/biocypher-out && chown 7474:7474 /serve/biocypher-out && chmod 755 /serve/biocypher-out'
mem_mb=$((SLURM_MEM_PER_NODE))
if [ "$TORCHCELL_PIP_REF" = "src" ]; then
    # pip cannot build from a read-only bind mount (it writes egg-info into the source
    # tree), so build the wheel on the host and hand the container the wheel.
    WHEEL_DIR="$BUILD_ROOT/database/slurm/output/${JOB}_wheel_${DATASET_CLASS}"
    mkdir -p "$WHEEL_DIR"
    python -m pip wheel --no-deps --no-build-isolation -w "$WHEEL_DIR" "$TORCHCELL_SRC" >/dev/null
    WHEEL=$(ls "$WHEEL_DIR"/torchcell-*.whl | head -1)
    [ -n "$WHEEL" ] || { echo "no wheel built from $TORCHCELL_SRC"; exit 1; }
    chmod -R a+rX "$WHEEL_DIR"
    SRC_MOUNT=(-v "$WHEEL_DIR":/mnt/torchcell-wheel:ro)
    PIP_TARGET="/mnt/torchcell-wheel/$(basename "$WHEEL")"
    echo "installing the worktree wheel $WHEEL"
else
    SRC_MOUNT=()
    PIP_TARGET="$TORCHCELL_PIP_REF"
fi
GEN_LOG="$BUILD_ROOT/database/slurm/output/${JOB}_generate_${DATASET_CLASS}.log"
docker run --rm --entrypoint bash \
    --cpus="$SLURM_CPUS_PER_TASK" -e SLURM_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK" --memory="${mem_mb}m" \
    -e NEO4J_ACCEPT_LICENSE_AGREEMENT=yes \
    -v "$SERVE_ROOT/biocypher-out":/var/lib/neo4j/biocypher-out \
    -v "$BUILD_ROOT/database/data":/var/lib/neo4j/data \
    -v "$BUILD_ROOT/database/.env":/.env \
    -v "$BUILD_ROOT/database/biocypher":/var/lib/neo4j/biocypher \
    "${SRC_MOUNT[@]}" \
    "$IMAGE" -c "set -euo pipefail
        cd /var/lib/neo4j
        python -m pip install --no-cache-dir --force-reinstall --no-deps '$PIP_TARGET' >/dev/null
        python -m pip install --no-cache-dir biocypher==0.15.2 >/dev/null
        set -a; source /.env; set +a
        wandb login \$WANDB_API_KEY >/dev/null
        python -m torchcell.knowledge_graphs.create_scerevisiae_kg_small --config-name kg_increment 'datasets=[$DATASET_CLASS]'
        # this container runs as root (pip needs the image's site-packages); hand the
        # output to the neo4j user, which runs the import, and keep it host-readable
        OUT_DIR=\$(dirname \$(cat biocypher_file_name.txt))
        chown -R 7474:7474 /var/lib/neo4j/\$OUT_DIR && chmod -R a+rX /var/lib/neo4j/\$OUT_DIR
        echo INCREMENT_CALL=\$(cat biocypher_file_name.txt)" 2>&1 | tee "$GEN_LOG"
CALL_REL=$(grep -oE '^INCREMENT_CALL=.*' "$GEN_LOG" | tail -1 | cut -d= -f2-)
[ -n "$CALL_REL" ] || { echo "no incremental call script reported"; exit 1; }
OUT_TS=$(basename "$(dirname "$CALL_REL")")
OUT_HOST="$SERVE_ROOT/biocypher-out/$OUT_TS"
OUT_CONT="/var/lib/neo4j/biocypher-out/$OUT_TS"
echo "increment CSVs: $OUT_HOST"
python - "$OUT_HOST/incremental-reference-analysis.json" <<'EOF'
import json, sys
a = json.load(open(sys.argv[1]))
print("node rows per label:", a["node_counts"]); print("edge rows per type:", a["edge_counts"])
print("distinct node ids", a["n_node_ids"], "| external endpoint ids", a["n_external_ids"], "| edges between existing nodes", a["n_edges_between_external"])
assert a["n_edges_between_external"] == 0
EOF
N_EXPECTED=$(python -c "import json; print(json.load(open('$OUT_HOST/incremental-reference-analysis.json'))['node_counts']['Experiment'])")

echo "---------------- 5 IMPORT into $NEO4J_CONTAINER/$NEO4J_DATABASE"
BEFORE=$(docker exec "$NEO4J_CONTAINER" $CS -d "$NEO4J_DATABASE" --format plain "MATCH (n) RETURN count(n);" | tail -1)
echo "nodes before: $BEFORE"
echo "-- constraints (database temporarily writable via server.databases.writable)"
docker exec "$NEO4J_CONTAINER" $CS -d system "CALL dbms.setConfigValue('server.databases.writable', '$NEO4J_DATABASE');"
docker exec "$NEO4J_CONTAINER" $CS -d "$NEO4J_DATABASE" -f "$OUT_CONT/incremental-constraints.cypher"
for i in $(seq 1 720); do
    NOT_ONLINE=$(docker exec "$NEO4J_CONTAINER" $CS -d "$NEO4J_DATABASE" --format plain \
        "SHOW INDEXES YIELD name, state WHERE state <> 'ONLINE' RETURN count(*);" | tail -1)
    [ "$NOT_ONLINE" = "0" ] && break
    echo "  $NOT_ONLINE constraint indexes still populating ($((i*10))s)"; sleep 10
done
docker exec "$NEO4J_CONTAINER" $CS -d "$NEO4J_DATABASE" "SHOW CONSTRAINTS YIELD name, labelsOrTypes, properties RETURN name, labelsOrTypes, properties;"
docker exec "$NEO4J_CONTAINER" $CS -d system "CALL dbms.setConfigValue('server.databases.writable', '');"
docker exec "$NEO4J_CONTAINER" $CS -d system "SHOW DATABASE $NEO4J_DATABASE YIELD name, access;"
echo "-- STOP DATABASE, incremental import as neo4j (uid 7474), START DATABASE"
docker exec "$NEO4J_CONTAINER" $CS -d system "STOP DATABASE $NEO4J_DATABASE WAIT;"
set +e
# HEAP_SIZE bounds neo4j-admin's JVM (it otherwise sizes itself from the whole machine)
# because it runs beside the still-running server inside the container's memory limit.
docker exec -u neo4j -e HEAP_SIZE=16G "$NEO4J_CONTAINER" bash "$OUT_CONT/neo4j-admin-incremental-import-call.sh"
IMPORT_RC=$?
set -e
docker exec "$NEO4J_CONTAINER" $CS -d system "START DATABASE $NEO4J_DATABASE WAIT;"
if [ "$IMPORT_RC" != "0" ]; then
    echo "INCREMENTAL IMPORT FAILED (rc=$IMPORT_RC); database restarted. Inspect $OUT_HOST/incremental-import.report"
    exit "$IMPORT_RC"
fi

echo "---------------- 6 VERIFY"
AFTER=$(docker exec "$NEO4J_CONTAINER" $CS -d "$NEO4J_DATABASE" --format plain "MATCH (n) RETURN count(n);" | tail -1)
N_LIVE=$(docker exec "$NEO4J_CONTAINER" $CS -d "$NEO4J_DATABASE" --format plain \
    "MATCH (d:Dataset {id:'$DATASET_CLASS'})<-[:ExperimentMemberOf]-(e:Experiment) RETURN count(e);" | tail -1)
echo "nodes before $BEFORE, after $AFTER (+$((AFTER - BEFORE))); $DATASET_CLASS experiments live: $N_LIVE, expected: $N_EXPECTED"
[ "$N_LIVE" = "$N_EXPECTED" ] || { echo "experiment count mismatch"; exit 1; }
cat "$OUT_HOST/incremental-import.report" 2>/dev/null | head -20 || true

echo "---------------- 7 RECORD + ARCHIVE"
python -m torchcell.knowledge_graphs.kg_manifest --manifest "$KG_MANIFEST" --repo "$TORCHCELL_SRC" \
    record --report "$REPORT" --biocypher-out "$OUT_TS" --n-experiments "$N_LIVE"
docker run --rm --entrypoint bash --user 7474:7474 -v "$SERVE_ROOT/biocypher-out":/src:ro -v "$ARCHIVE_ROOT":/dst "$IMAGE" \
    -c "cp -a /src/$OUT_TS /dst/ && ls -ld /dst/$OUT_TS"
echo "DONE at $(date)"
