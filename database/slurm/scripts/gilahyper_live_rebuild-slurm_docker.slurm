#!/bin/bash
#SBATCH --job-name=live_rebuild_kg
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=4-00:00:00
#SBATCH --output=/scratch/projects/torchcell/database/slurm/output/%j_%x.out
#SBATCH --error=/scratch/projects/torchcell/database/slurm/output/%j_%x.out
#SBATCH --no-requeue

# The LIVE full rebuild: build a complete new store while the served one keeps serving,
# then swap the two by renaming directories.
#
#   sbatch database/slurm/scripts/gilahyper_live_rebuild-slurm_docker.slurm
#
# HOW IT DIFFERS from gilahyper_uncapped_build-slurm_docker.slurm, which stops the
# serving container first and rebuilds in place (a day of downtime, and no way back):
#
#   generate  an ephemeral python container (--entrypoint bash, the pattern of the
#             incremental runner) reads the DEV-tree LMDBs read-only -- the stores L0-L4
#             verification ran on, not staged copies -- and writes CSVs into
#             $SERVE_ROOT/biocypher-out/<ts>, the directory the serving container already
#             mounts, so a later incremental admission finds them. No neo4j runs in it:
#             the image entrypoint chowns NEO4J_HOME recursively, which a read-only
#             mount under /var/lib/neo4j/data would turn into a restart loop.
#   import    a build container (tc-neo4j-build, NO host ports) runs neo4j-admin import +
#             CREATE DATABASE against a FRESH neo4j data root, $NEXT_ROOT/data, on the
#             same /db filesystem as the served store.
#   validate  the new store is queried before anything served is touched: every mapped
#             dataset has a Dataset node and its live ExperimentMemberOf count equals its
#             CSV rows.
#   swap      stop both containers; mv $SERVE_ROOT/data -> data.superseded.<ts>;
#             mv $NEXT_ROOT/data -> $SERVE_ROOT/data; relaunch tc-neo4j-readonly with the
#             arguments scripts/migrate_storage_tiers.sh uses. Downtime is the two renames
#             plus one neo4j start (minutes), and the old store stays on disk until
#             someone deletes it: rollback is the reverse pair of mv.
#   record    kg_manifest bootstrap from the live store (the old manifest is kept beside
#             it), CSVs archived to /bulk.
#
# RESOURCES. 32 CPUs and 200 G: MaxMemPerCPU=4100 turns 200 G into a 50-CPU floor, so
# with the four 4-CPU/60 G GPU jobs that usually run the node keeps 62 CPUs and ~70 G
# free; 320 G (the uncapped script) would not start beside them. The build's measured
# peak private memory with the gc.freeze fix is ~41 G at 29 workers (kg_uncapped.yaml);
# the rest is page cache, which the kernel reclaims. neo4j-admin sizes its heap and
# off-heap from the container limit and imported job 1558's 83 M nodes in 19 min at
# 320 G.
#
# DISK. Job 1558 wrote 526 G of CSV for 36 datasets; the new store is ~535 G. Both land
# on /db (the CSVs under $SERVE_ROOT/biocypher-out, the store under $NEXT_ROOT), so /db
# needs ~1.2 T free on top of the old store, which stays until the superseded copy is
# removed. The preflight refuses below MIN_FREE_GB (1500).

source ~/miniconda3/etc/profile.d/conda.sh
conda activate torchcell
set -euo pipefail

KG_CONFIG="${KG_CONFIG:-kg_uncapped}"
KG_MODULE="${KG_MODULE:-torchcell.knowledge_graphs.create_scerevisiae_kg_small}"
TORCHCELL_SRC="${TORCHCELL_SRC:-/home/michaelvolk/Documents/projects/torchcell}"
DEV_DATA_ROOT="${DEV_DATA_ROOT:-/scratch/projects/torchcell-scratch}"
BUILD_ROOT="${BUILD_ROOT:-/scratch/projects/torchcell}"
SERVE_ROOT="${SERVE_ROOT:-/db/database}"
NEXT_ROOT="${NEXT_ROOT:-/db/database-next}"
NEO4J_CONTAINER="${NEO4J_CONTAINER:-tc-neo4j-readonly}"
BUILD_CONTAINER="${BUILD_CONTAINER:-tc-neo4j-build}"
GEN_CONTAINER="${GEN_CONTAINER:-tc-neo4j-gen}"
IMAGE="${IMAGE:-michaelvolk/tc-neo4j:5.26.28}"
ARCHIVE_ROOT="${ARCHIVE_ROOT:-/bulk/biocypher-out}"
KG_MANIFEST="${KG_MANIFEST:-$BUILD_ROOT/database/kg_manifest.json}"
SERVE_CPUS="${SERVE_CPUS:-16}"
SERVE_MEM="${SERVE_MEM:-200g}"
MIN_FREE_GB="${MIN_FREE_GB:-1500}"
DATABASE_ENV_FILE="${DATABASE_ENV_FILE:-/home/michaelvolk/Documents/projects/torchcell/database/database.env}"
NEO4J_DATABASE=torchcell
JOB="${SLURM_JOB_ID:-nojob}"
OUTDIR="$BUILD_ROOT/database/slurm/output"
mem_mb=$((SLURM_MEM_PER_NODE))
CS='cypher-shell -u neo4j -p torchcell'
# The container installs torchcell from GitHub at this commit; the manifest records it.
git -C "$TORCHCELL_SRC" fetch origin main --quiet
BUILD_COMMIT="${BUILD_COMMIT:-$(git -C "$TORCHCELL_SRC" rev-parse origin/main)}"
PIP_REF="git+https://github.com/Mjvolk3/torchcell.git@$BUILD_COMMIT"

echo "Running on $(hostname) at $(date); job $JOB"
echo "config $KG_CONFIG; commit $BUILD_COMMIT; dev tree $DEV_DATA_ROOT; serve $SERVE_ROOT; next $NEXT_ROOT"
echo "SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK SLURM_MEM_PER_NODE=$SLURM_MEM_PER_NODE"
cd "$TORCHCELL_SRC"
export PYTHONPATH="$TORCHCELL_SRC"

echo "---------------- 0 PREFLIGHT"
# The fingerprint repo must carry the same code the container installs.
if ! git -C "$TORCHCELL_SRC" diff --quiet "$BUILD_COMMIT" HEAD -- torchcell biocypher; then
    echo "$TORCHCELL_SRC HEAD differs from $BUILD_COMMIT under torchcell/ or biocypher/"; exit 1
fi
docker inspect "$NEO4J_CONTAINER" --format '{{.State.Status}}' | grep -qx running \
    || { echo "$NEO4J_CONTAINER is not running"; exit 1; }
for c in "$BUILD_CONTAINER" "$GEN_CONTAINER"; do
    if docker inspect "$c" >/dev/null 2>&1; then echo "$c already exists; remove it first"; exit 1; fi
done
BEFORE=$(docker exec "$NEO4J_CONTAINER" $CS -d "$NEO4J_DATABASE" --format plain "MATCH (n) RETURN count(n);" | tail -1)
BEFORE_DS=$(docker exec "$NEO4J_CONTAINER" $CS -d "$NEO4J_DATABASE" --format plain "MATCH (d:Dataset) RETURN count(d);" | tail -1)
echo "served now: $BEFORE nodes, $BEFORE_DS datasets"
FREE_GB=$(df --output=avail -BG /db | tail -1 | tr -dc 0-9)
[ "$FREE_GB" -ge "$MIN_FREE_GB" ] || { echo "/db has ${FREE_GB}G free, need ${MIN_FREE_GB}G"; exit 1; }
if [ -e "$NEXT_ROOT/data" ] && [ -n "$(ls -A "$NEXT_ROOT/data")" ]; then
    echo "$NEXT_ROOT/data exists and is not empty; move it aside first"; exit 1
fi
# Every mapped dataset must have a dev LMDB whose build manifest is fresh under the
# schema the container will install: the store is built from exactly those records.
N_DATASETS=$(python - "$DEV_DATA_ROOT" <<'EOF'
import os, sys
from torchcell.knowledge_graphs.dataset_adapter_map import dataset_adapter_map
from torchcell.database.build_dataset_lmdb import dataset_default_root
from torchcell.provenance.build_manifest import BuildManifest, check_manifest
from torchcell.provenance.schema_deps import load_default_surface
dev = sys.argv[1]
surface = load_default_surface()
bad = []
for cls in dataset_adapter_map:
    root = os.path.join(dev, dataset_default_root(cls))
    pre = os.path.join(root, "preprocess")
    mp = os.path.join(pre, "build_manifest.json")
    if not os.path.isdir(os.path.join(root, "processed", "lmdb")):
        bad.append(f"{cls.__name__}: no LMDB at {root}"); continue
    if not os.path.isfile(mp):
        bad.append(f"{cls.__name__}: no build manifest"); continue
    res = check_manifest(BuildManifest.model_validate_json(open(mp).read()), surface, pre)
    if res.is_stale:
        bad.append(f"{cls.__name__}: stale on {sorted({d.symbol for d in res.drift})}")
if bad:
    print("\n".join(bad), file=sys.stderr); sys.exit(1)
print(len(dataset_adapter_map))
EOF
)
echo "$N_DATASETS mapped datasets, every dev store fresh"
NEXT_PARENT=$(dirname "$NEXT_ROOT"); NEXT_NAME=$(basename "$NEXT_ROOT")
docker run --rm --entrypoint bash -v "$NEXT_PARENT":/parent "$IMAGE" -c \
    "mkdir -p /parent/$NEXT_NAME/data /parent/$NEXT_NAME/logs /parent/$NEXT_NAME/metrics \
     && chown -R 7474:7474 /parent/$NEXT_NAME && ls -la /parent/$NEXT_NAME"
docker run --rm --entrypoint bash -v "$SERVE_ROOT":/serve "$IMAGE" -c \
    'mkdir -p /serve/biocypher-out && chown 7474:7474 /serve/biocypher-out && chmod 755 /serve/biocypher-out'
# build-tree biocypher config + container env come from the checkout (schema must match)
DATA_ROOT="$BUILD_ROOT" python -m torchcell.database.directory_setup --env-file "$DATABASE_ENV_FILE"
cmp "$TORCHCELL_SRC/biocypher/config/torchcell_schema_config.yaml" \
    "$BUILD_ROOT/database/biocypher/config/torchcell_schema_config.yaml"
df -h /db /scratch

echo "---------------- 1 GENERATE (ephemeral container; dev tree read-only) $(date)"
# The dev tree's data/torchcell shadows the build tree's copy read-only (loaders open
# LMDB readonly, lock=False); genome/graph caches (data/sgd, go, string, tflink) come
# from the build tree as in every previous build. The container runs as root so pip can
# install into the image; the CSVs are handed to uid 7474, which runs the import.
GEN_LOG="$OUTDIR/${JOB}_generate.log"
docker run --rm --entrypoint bash --name "$GEN_CONTAINER" \
    --cpus="$SLURM_CPUS_PER_TASK" -e SLURM_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK" --memory="${mem_mb}m" \
    -e NEO4J_ACCEPT_LICENSE_AGREEMENT=yes \
    -v "$SERVE_ROOT/biocypher-out":/var/lib/neo4j/biocypher-out \
    -v "$BUILD_ROOT/database/data":/var/lib/neo4j/data \
    -v "$DEV_DATA_ROOT/data/torchcell":/var/lib/neo4j/data/torchcell:ro \
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
        python -m $KG_MODULE --config-name $KG_CONFIG
        CALL=\$(cat biocypher_file_name.txt)
        OUT_DIR=\$(dirname \$CALL)
        chown -R 7474:7474 /var/lib/neo4j/\$OUT_DIR && chmod -R a+rX /var/lib/neo4j/\$OUT_DIR
        echo GENERATION_CALL=\$CALL" 2>&1 | tee "$GEN_LOG"
echo "CSV generation completed at $(date)"
N_ADAPTERS=$(grep -oE 'across [0-9]+ adapters' "$GEN_LOG" | tail -1 | grep -oE '[0-9]+')
[ "$N_ADAPTERS" = "$N_DATASETS" ] || { echo "generation covered '$N_ADAPTERS' adapters, expected $N_DATASETS"; exit 1; }
CALL_REL=$(grep -oE '^GENERATION_CALL=.*' "$GEN_LOG" | tail -1 | cut -d= -f2-)
[ -n "$CALL_REL" ] || { echo "no import call reported"; exit 1; }
OUT_TS=$(basename "$(dirname "$CALL_REL")")
OUT_HOST="$SERVE_ROOT/biocypher-out/$OUT_TS"
OUT_CONT="/var/lib/neo4j/biocypher-out/$OUT_TS"
echo "CSVs: $OUT_HOST ($CALL_REL)"
df -h /db

echo "---------------- 2 IMPORT into $NEXT_ROOT ($BUILD_CONTAINER, no host ports) $(date)"
docker run \
    --cpus="$SLURM_CPUS_PER_TASK" --memory="${mem_mb}m" \
    --env=NEO4J_ACCEPT_LICENSE_AGREEMENT=yes \
    -d --name "$BUILD_CONTAINER" --restart=no \
    -v "$NEXT_ROOT/data":/var/lib/neo4j/data \
    -v "$SERVE_ROOT/biocypher-out":/var/lib/neo4j/biocypher-out \
    -v "$BUILD_ROOT/database/.env":/.env \
    -v "$BUILD_ROOT/database/biocypher":/var/lib/neo4j/biocypher \
    -v "$BUILD_ROOT/database/conf":/var/lib/neo4j/conf \
    -v "$NEXT_ROOT/logs":/logs \
    -v "$NEXT_ROOT/metrics":/metrics \
    -e NEO4J_AUTH=neo4j/torchcell \
    -e NEO4J_server_databases_default__to__read__only=false \
    "$IMAGE"
sleep 60
docker ps -q -f name="$BUILD_CONTAINER" | grep -q . || { docker logs "$BUILD_CONTAINER"; echo "$BUILD_CONTAINER failed to start"; exit 1; }
docker exec "$BUILD_CONTAINER" $CS -d system "DROP DATABASE $NEO4J_DATABASE IF EXISTS;"
# as uid neo4j so the store files are neo4j-owned (a root import leaves files the
# server cannot open and the database goes DIRTY)
docker exec -u neo4j "$BUILD_CONTAINER" bash "$OUT_CONT/neo4j-admin-import-call.sh"
docker exec "$BUILD_CONTAINER" bash -c 'chown -R neo4j:neo4j /var/lib/neo4j/data/databases /var/lib/neo4j/data/transactions'
docker exec "$BUILD_CONTAINER" $CS -d system "CREATE DATABASE $NEO4J_DATABASE;"
for i in $(seq 1 180); do
    STATUS=$(docker exec "$BUILD_CONTAINER" $CS -d system --format plain \
        "SHOW DATABASE $NEO4J_DATABASE YIELD currentStatus RETURN currentStatus;" | tail -1 | tr -d '"')
    [ "$STATUS" = "online" ] && break
    echo "  $NEO4J_DATABASE is $STATUS ($((i*10))s)"; sleep 10
done
[ "$STATUS" = "online" ] || { echo "$NEO4J_DATABASE never came online in $BUILD_CONTAINER"; exit 1; }
# the index-stats file the read-only container needs is written on a stop/start + query
docker exec "$BUILD_CONTAINER" neo4j stop
docker exec "$BUILD_CONTAINER" neo4j start
sleep 60
for i in $(seq 1 60); do
    STATUS=$(docker exec "$BUILD_CONTAINER" $CS -d system --format plain \
        "SHOW DATABASE $NEO4J_DATABASE YIELD currentStatus RETURN currentStatus;" 2>/dev/null | tail -1 | tr -d '"')
    [ "$STATUS" = "online" ] && break
    sleep 10
done
NEW_COUNT=$(docker exec "$BUILD_CONTAINER" $CS -d "$NEO4J_DATABASE" --format plain "MATCH (n) RETURN count(n);" | tail -1)
echo "imported store: $NEW_COUNT nodes (served store: $BEFORE)"

echo "---------------- 3 VALIDATE before touching the served store"
VALIDATION="$OUTDIR/${JOB}_validation.txt"
NEW_DS=$(docker exec "$BUILD_CONTAINER" $CS -d "$NEO4J_DATABASE" --format plain "MATCH (d:Dataset) RETURN count(d);" | tail -1)
echo "Dataset nodes: $NEW_DS (expected $N_DATASETS)" | tee "$VALIDATION"
[ "$NEW_DS" = "$N_DATASETS" ] || { echo "dataset count mismatch"; exit 1; }
# ExperimentMemberOf rows per Dataset in the CSVs are what the store must hold
EXPECTED="$OUTDIR/${JOB}_expected.txt"
python - "$OUT_HOST" > "$EXPECTED" <<'EOF'
import csv, sys
from collections import Counter
from pathlib import Path
from torchcell.knowledge_graphs.incremental_import import discover_csv_groups
csv.field_size_limit(sys.maxsize)
out_dir = Path(sys.argv[1])
group = next(g for g in discover_csv_groups(out_dir) if g.label == "ExperimentMemberOf")
end = group.columns.index(":END_ID")
counts: Counter[str] = Counter()
for part in group.part_paths:
    with open(part, encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle, delimiter="\t", quotechar="'", strict=True):
            counts[row[end].strip("'")] += 1
for name, n in sorted(counts.items()):
    print(f"{name}={n}")
EOF
docker exec "$BUILD_CONTAINER" $CS -d "$NEO4J_DATABASE" --format plain \
    "MATCH (d:Dataset)<-[:ExperimentMemberOf]-(e:Experiment) RETURN d.id + '=' + toString(count(e)) AS row ORDER BY row;" \
    | tail -n +2 | tr -d '"' | sort > "$OUTDIR/${JOB}_live.txt"
if diff "$EXPECTED" "$OUTDIR/${JOB}_live.txt" | tee -a "$VALIDATION"; then
    echo "per-dataset experiment counts: live == CSV for all $N_DATASETS datasets" | tee -a "$VALIDATION"
else
    echo "per-dataset experiment counts differ between the CSVs and the store"; exit 1
fi
echo "import.report (first 20 lines):" | tee -a "$VALIDATION"
docker exec "$BUILD_CONTAINER" bash -c "f=\$(ls $OUT_CONT/import.report /var/lib/neo4j/import.report /var/lib/neo4j/data/import.report 2>/dev/null | head -1); [ -n \"\$f\" ] && head -20 \"\$f\" || echo '(no import.report found)'" | tee -a "$VALIDATION"
echo "total nodes: $NEW_COUNT (served before: $BEFORE)" | tee -a "$VALIDATION"

echo "---------------- 4 SWAP $(date)"
docker stop "$BUILD_CONTAINER"
docker rm "$BUILD_CONTAINER"
SWAP_TS=$(date +%Y-%m-%d_%H-%M-%S)
SERVE_PARENT=$(dirname "$SERVE_ROOT"); SERVE_NAME=$(basename "$SERVE_ROOT")
docker stop "$NEO4J_CONTAINER"
docker rm "$NEO4J_CONTAINER"
docker run --rm --entrypoint bash -v "$SERVE_PARENT":/serve_parent -v "$NEXT_PARENT":/next_parent "$IMAGE" -c "set -e
    mv /serve_parent/$SERVE_NAME/data /serve_parent/$SERVE_NAME/data.superseded.$SWAP_TS
    mv /next_parent/$NEXT_NAME/data /serve_parent/$SERVE_NAME/data
    ls -la /serve_parent/$SERVE_NAME /serve_parent/$SERVE_NAME/data/databases"
echo "old store kept at $SERVE_ROOT/data.superseded.$SWAP_TS (rollback: stop the container, reverse the two mv, relaunch)"
docker run \
    --cpus="$SERVE_CPUS" \
    --memory="$SERVE_MEM" \
    --env=NEO4J_ACCEPT_LICENSE_AGREEMENT=yes \
    -d --name "$NEO4J_CONTAINER" \
    -p 7687:7687 -p 7474:7474 \
    --restart=unless-stopped \
    -v "$SERVE_ROOT/data":/var/lib/neo4j/data \
    -v "$SERVE_ROOT/biocypher-out":/var/lib/neo4j/biocypher-out \
    -v "$SERVE_ROOT/.env":/.env \
    -v "$SERVE_ROOT/biocypher":/var/lib/neo4j/biocypher \
    -v "$SERVE_ROOT/conf":/var/lib/neo4j/conf \
    -v "$SERVE_ROOT/logs":/logs \
    -v "$SERVE_ROOT/metrics":/metrics \
    -e NEO4J_AUTH=neo4j/torchcell \
    -e NEO4J_server_databases_default__to__read__only=true \
    "$IMAGE"
for i in $(seq 1 120); do
    STATUS=$(docker exec "$NEO4J_CONTAINER" $CS -d system --format plain \
        "SHOW DATABASE $NEO4J_DATABASE YIELD currentStatus RETURN currentStatus;" 2>/dev/null | tail -1 | tr -d '"')
    [ "$STATUS" = "online" ] && break
    echo "  serving $NEO4J_DATABASE is ${STATUS:-starting} ($((i*10))s)"; sleep 10
done
[ "$STATUS" = "online" ] || { echo "served $NEO4J_DATABASE did not come online; see docker logs $NEO4J_CONTAINER"; exit 1; }
SERVED=$(docker exec "$NEO4J_CONTAINER" $CS -d "$NEO4J_DATABASE" --format plain "MATCH (n) RETURN count(n);" | tail -1)
echo "SERVING $SERVED nodes (validated $NEW_COUNT) at $(date)"
[ "$SERVED" = "$NEW_COUNT" ] || { echo "served count differs from the validated store"; exit 1; }
docker exec "$NEO4J_CONTAINER" $CS -d system "SHOW DATABASES YIELD name, currentStatus, access;"

echo "---------------- 5 RECORD + ARCHIVE"
if [ -f "$KG_MANIFEST" ]; then
    mv "$KG_MANIFEST" "$KG_MANIFEST.superseded.$SWAP_TS"
    echo "previous manifest kept at $KG_MANIFEST.superseded.$SWAP_TS"
fi
python -m torchcell.knowledge_graphs.kg_manifest --manifest "$KG_MANIFEST" --repo "$TORCHCELL_SRC" \
    bootstrap --commit "$BUILD_COMMIT" --biocypher-out "$OUT_TS" --built-at "$(date -Iseconds)" \
    --biocypher-version 0.15.2 --database "$NEO4J_DATABASE" --neo4j-uri bolt://localhost:7687
python -m torchcell.knowledge_graphs.kg_manifest --manifest "$KG_MANIFEST" --repo "$TORCHCELL_SRC" show
docker run --rm --entrypoint bash --user 7474:7474 -v "$SERVE_ROOT/biocypher-out":/src:ro -v "$ARCHIVE_ROOT":/dst "$IMAGE" \
    -c "cp -a /src/$OUT_TS /dst/ && ls -ld /dst/$OUT_TS"
df -h /db /scratch /bulk
echo "DONE at $(date): $SERVED nodes, $N_DATASETS datasets served from $SERVE_ROOT/data; CSVs $OUT_HOST (archived to $ARCHIVE_ROOT/$OUT_TS); validation $VALIDATION"
