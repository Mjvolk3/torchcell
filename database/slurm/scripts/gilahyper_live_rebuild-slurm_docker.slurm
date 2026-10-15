#!/bin/bash
#SBATCH --job-name=live_rebuild_kg
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/projects/torchcell/database/slurm/output/%j_%x.out
#SBATCH --error=/scratch/projects/torchcell/database/slurm/output/%j_%x.out
#SBATCH --no-requeue

# The LIVE full rebuild: build a complete new store while the served one keeps serving,
# then swap the two by renaming directories.
#
#   sbatch database/slurm/scripts/gilahyper_live_rebuild-slurm_docker.slurm
#
# RESUME after a failure past the import. When a job dies after its import has
# completed (job 2032 did, at the post-import restart), the CSVs and the imported store
# are intact and generation (29 h) need not be repeated:
#
#   sbatch --export=ALL,RESUME_JOB=<old job id> database/slurm/scripts/gilahyper_live_rebuild-slurm_docker.slurm
#
# The resume reads the old job's logs for the generation commit, the CSV directory, and
# the dataset count, requires the build container to exist and be stopped with the
# imported database under $NEXT_ROOT/data, then starts the container and continues from
# the restart: count, validate, swap, record. The manifest is bootstrapped at the
# generation commit (kg_manifest reads the schema surface with git show at that ref), so
# the checkout may be ahead of it.
#
# RESUME after a failure past the generation. When the job dies between the end of
# generation and the end of the import (job 4901 hit its 12 h time limit at 10:27 CDT on
# 2026-10-14 with the generation container still running), the CSVs are complete or
# completing and nothing has been imported:
#
#   sbatch --export=ALL,RESUME_JOB=<old job id>,RESUME_FROM=generation database/slurm/scripts/gilahyper_live_rebuild-slurm_docker.slurm
#
# The resume reads the old job's logs for the commit, the organism set and the dataset
# count as above, requires its generate log to end in GENERATION_CALL= (the generator
# finished and handed the CSVs to uid 7474), re-digests the dev tree against the old
# job's fence, then imports, validates, swaps and records as a fresh run does. A job
# that dies by time limit does NOT stop its generation container: docker runs it
# outside the job's cgroup, and `--rm` removes it, and its log, the moment it exits.
# Keep the rest of its output before that happens, then append it to the generate log
# the resume reads (same stream, captured by a different process):
#
#   nohup docker logs -f --since <UTC time the job died>Z tc-neo4j-gen > <OUTDIR>/<job>_generate_after_timeout.log 2>&1 &
#   ... wait for the container to exit ...
#   cat <OUTDIR>/<job>_generate_after_timeout.log >> <OUTDIR>/<job>_generate.log
#
# `--since` without the trailing Z is read as local time and captures nothing.
#
# GENERATION_MODE=array (2026-10-15; the default stays serial until the owner flips it).
# Generation runs as ONE SLURM ARRAY TASK PER ADAPTER
# (database/slurm/scripts/gilahyper_generate_adapter_array-slurm_docker.slurm, one
# container tc-neo4j-gen-<i> each, one directory biocypher-out/<ts>/adapters/<iii>_<Adapter>/
# each, a DONE marker written last) instead of one serial container, so a failure at
# adapter 124 costs adapter 124, not adapters 1 to 123. This job then:
#
#   1. preflights and fences as a serial run does, starts the array through
#      database/scripts/kg_generation_array_conductor.sh (one sbatch per resource tier,
#      torchcell/knowledge_graphs/conf/generation_tiers.yaml; the fence is recorded at
#      biocypher-out/<ts>/dev_fence.txt and every task re-checks it), prints
#      ARRAY_OUT=<ts> and ARRAY_JOBS=<ids>,
#   2. submits ITSELF again with --dependency=afterany:<ids> and
#      RESUME_JOB=<this job> RESUME_FROM=array ARRAY_FOLLOWUP=1, and exits, so the node
#      is not held by a waiting job while the array runs,
#   3. the follow-up verifies every directory carries DONE (refusing with the exact
#      re-run command when one does not), re-digests the fence, MERGES the directories
#      into one neo4j-admin-import-call.sh (torchcell.knowledge_graphs.generation_array
#      merge: per label one header and the comma-joined part files; duplicate
#      content-id nodes proven byte-identical and skipped by --skip-duplicate-nodes;
#      duplicate edges dropped into rewritten part files), then passes the CSV endpoint
#      gate the import branch runs in every mode (torchcell.knowledge_graphs.csv_endpoint_gate
#      walks adapters/*/ as one output), imports, validates, swaps and records.
#
# RECOVERY in array mode: fix the cause, then
#
#   sbatch --export=ALL,RESUME_JOB=<array-mode job>,RESUME_FROM=array database/slurm/scripts/gilahyper_live_rebuild-slurm_docker.slurm
#
# re-runs exactly the tasks without a DONE marker (the other directories are untouched)
# and chains a new follow-up. A follow-up itself never re-runs a task: it refuses and
# prints the command, so a failure that recurs cannot loop. In array mode the job that
# imports needs the import container's size, not a generation container's:
# IMPORT_CPUS / IMPORT_MEM_MB (default: this job's SLURM_CPUS_PER_TASK / SLURM_MEM_PER_NODE).
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
#             same /db filesystem as the served store, then makes torchcell the default
#             database of that store (its system database moves with it in the swap).
#   validate  the new store is queried before anything served is touched: every mapped
#             dataset has a Dataset node and its live ExperimentMemberOf count equals its
#             CSV rows.
#   swap      stop both containers; mv $SERVE_ROOT/data -> data.superseded.<ts>;
#             mv $NEXT_ROOT/data -> $SERVE_ROOT/data; relaunch tc-neo4j-readonly with the
#             arguments scripts/migrate_storage_tiers.sh uses. Downtime is the two renames
#             plus one neo4j start (minutes), and the old store stays on disk until
#             someone deletes it: rollback is the reverse pair of mv.
#   record    kg_manifest bootstrap from the live store (the old manifest is kept beside
#             it), kg_manifest artifact-refs (the distinct artifact files each dataset's
#             records point at, read from the dev-tree LMDBs), the release stamped
#             (version, release id, per-dataset content hashes, the pointer set) into
#             the manifest and the store's KgRelease node, the latest and
#             pinned aliases created, CSVs archived to /bulk. The release artifact
#             itself (an online backup) is scripts/kg_release.sh, run afterwards.
#
# PRIVATE DATASETS. The GilaHyper store is the in-house store, so the build includes the
# private datasets (PRIVATE_DATASET_ADAPTER_MAP, visibility=private) by default:
# INCLUDE_PRIVATE=1 passes --include-private to the generator and counts the private map
# in the preflight. INCLUDE_PRIVATE=0 builds the public datasets alone. A tc-data
# release refuses a private dataset either way (scripts/package_dataset_lmdb.py).
#
# ORGANISMS. KG_ORGANISMS (default scerevisiae,ecoli,pputida) is the organism set the
# store serves: a comma-separated subset of the Organism enum
# (torchcell/data/experiment_dataset.py), validated and put in enum order before
# anything runs. It narrows ONE map everywhere: the preflight's freshness walk and
# dataset count, the dev fence, and the generator (--organisms), so a yeast-only build
#
#   sbatch --export=ALL,KG_ORGANISMS=scerevisiae database/slurm/scripts/gilahyper_live_rebuild-slurm_docker.slurm
#
# neither blocks on a stale bacterial store nor expects a bacterial Dataset node. The
# set is written into kg_manifest.json (bootstrap --organisms), and from there into the
# release snapshot and the KgRelease node, so the store says it is yeast-only. A RESUME
# must name the same set the resumed job generated (its log line "organisms ..."). The
# set is one word of the sbatch --export list only when it has one member: sbatch splits
# that list on commas, so a set of two or more is exported from the caller's
# environment (KG_ORGANISMS=scerevisiae,ecoli sbatch --export=ALL ...).
#
# RESOURCES. 64 CPUs, 256 G, 48 h. The 12 h this header carried until 2026-10-14 was
# measured on 51 datasets; job 4901 (170 datasets, three organisms) was killed by it at
# 12 h 00 min inside the serial adapter pass, where Hoepfner 2014 alone (30.3 M
# experiments, 39 KB resolved per record) took 4 h 14 min of single-threaded node
# writing and about 2 h 15 min of edge writing, with the node 96% idle. 48 h is a
# ceiling for the job, not a reservation of anything but the slot: the generation ran
# 12 h to 106 of 170 adapters, and the import, validation and swap are an hour or
# two on top of the generation. SLURM_CPUS_PER_TASK sets the adapter worker
# count (get_num_workers in create_scerevisiae_kg_small), so it is a build parameter and
# not just a reservation; MaxMemPerCPU=4100 makes 256 G a 64-CPU floor in any case
# (ceil(262144 / 4100)), which is why 24 in this header got job 2032 allocated 48. Jobs
# 3198, 3297 and 3547 all overrode both directives on the command line, so the header is
# now what they ran with. 64 CPUs plus 256 G leaves exactly the 64 CPUs and 216,000 MB
# the four 16-CPU / 54,000 M GPU packs want on a 128-CPU / 513,000 MB node; 320 G forces
# an 80-CPU floor and leaves room for three.
#
# Memory does not scale with the dataset count: the pool recycles once the container's
# cgroup memory passes pool_memory_fraction (0.8) of its limit, so the peak rides the
# ceiling rather than needing it. Job 3297's 22 recycles fired between 0.80 and 0.88 and
# it closed at "Generation wall 11102 s, peak memory 235.4 GiB" for 51 datasets. What a
# bigger request would cover is the per-worker working set, a 128 MiB resolved-record
# chunk times the process_workers + 2 chunks in flight, which the config sets. Job 1996
# at 32 CPUs / 200 G was OOM-killed 15 min in with one worker holding 45 G, before the
# interning and pool-payload fixes; job 2032 at 256 G took 28 h 40 min for the same 51
# datasets those fixes brought to 3 h 05 min.
#
# TIME. 51 datasets: 3 h 11 min (job 3198) and 3 h 36 min (job 3297) end to end, of which
# generation was 3 h 05 min and neo4j-admin import 6 min 48 s. 109 datasets (the first
# build with the bacterial program) project to 4 h 15 min with a 6 h 30 min upper bound:
# the 58 new stores carry 4.02 M records and 17.6 GB of dev LMDB against 52.67 M and
# 205.2 GB already served. Projection and its rule:
# notes/database.kg-4-build-sizing.md, experiments/database/scripts/kg4_build_sizing.py.
#
# DISK. Job 3297 wrote ~200 G of CSV for 51 datasets and its imported store is 150 G
# (the pre-interning job 2032 wrote 634 G of CSV for a 682 G store). Both land on /db
# (the CSVs under $SERVE_ROOT/biocypher-out, the store under $NEXT_ROOT), and the old
# store stays as data.superseded.<ts> until someone removes it. 109 datasets project to
# 215 G of CSV and a 162 G store. The preflight refuses below MIN_FREE_GB (1500), which
# is ~4x the projected need and is there to keep the rollback copy affordable.

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
# The base image plus the Browser styling seed (database/docker/Dockerfile.tc-neo4j-browser).
IMAGE="${IMAGE:-michaelvolk/tc-neo4j:5.26.28-browser.1}"
ARCHIVE_ROOT="${ARCHIVE_ROOT:-/bulk/biocypher-out}"
# Release artifacts (online backups + manifests, scripts/kg_release.sh) live here and
# the serving container mounts it at /kg-releases, OUTSIDE NEO4J_HOME: the entrypoint
# chmods every top-level directory under NEO4J_HOME to 700 at each start, which would
# hide the archive from the host. Backups cannot stage on /db: a full backup holds the
# whole store (680 G) in its work directory before compressing.
RELEASE_ROOT="${RELEASE_ROOT:-/bulk/kg-releases}"
KG_MANIFEST="${KG_MANIFEST:-$BUILD_ROOT/database/kg_manifest.json}"
INCLUDE_PRIVATE="${INCLUDE_PRIVATE:-1}"
case "$INCLUDE_PRIVATE" in
    1) PRIVATE_FLAG="--include-private" ;;
    0) PRIVATE_FLAG="" ;;
    *) echo "INCLUDE_PRIVATE must be 0 or 1, got '$INCLUDE_PRIVATE'"; exit 1 ;;
esac
KG_ORGANISMS="${KG_ORGANISMS:-scerevisiae,ecoli,pputida}"
[[ "$KG_ORGANISMS" =~ ^(scerevisiae|ecoli|pputida)(,(scerevisiae|ecoli|pputida))*$ ]] \
    || { echo "KG_ORGANISMS must be a comma-separated subset of scerevisiae,ecoli,pputida, got '$KG_ORGANISMS'"; exit 1; }
ORGANISMS_CANONICAL=""
for ORGANISM in scerevisiae ecoli pputida; do
    if [[ ",$KG_ORGANISMS," == *",$ORGANISM,"* ]]; then
        ORGANISMS_CANONICAL="${ORGANISMS_CANONICAL:+$ORGANISMS_CANONICAL,}$ORGANISM"
    fi
done
KG_ORGANISMS="$ORGANISMS_CANONICAL"
SERVE_CPUS="${SERVE_CPUS:-16}"
SERVE_MEM="${SERVE_MEM:-200g}"
MIN_FREE_GB="${MIN_FREE_GB:-1500}"
DATABASE_ENV_FILE="${DATABASE_ENV_FILE:-/home/michaelvolk/Documents/projects/torchcell/database/database.env}"
NEO4J_DATABASE=torchcell
JOB="${SLURM_JOB_ID:-nojob}"
OUTDIR="$BUILD_ROOT/database/slurm/output"
mem_mb=$((SLURM_MEM_PER_NODE))
CS='cypher-shell -u neo4j -p torchcell'
RESUME_JOB="${RESUME_JOB:-}"
MERGE_REPORT=""
# RESUME_FROM names the last stage the resumed job finished: `import` (the default, the
# resume starts at the post-import restart) or `generation` (the resume starts at the
# import). GENERATE and IMPORT are what THIS job runs.
RESUME_FROM="${RESUME_FROM:-import}"
case "$RESUME_FROM" in
    import|generation|array) ;;
    *) echo "RESUME_FROM must be import, generation or array, got '$RESUME_FROM'"; exit 1 ;;
esac
# GENERATION_MODE: serial (one container, the default) or array (one task per adapter;
# see the header). A RESUME_FROM=array implies the array mode.
GENERATION_MODE="${GENERATION_MODE:-serial}"
[ "$RESUME_FROM" = array ] && GENERATION_MODE=array
case "$GENERATION_MODE" in
    serial|array) ;;
    *) echo "GENERATION_MODE must be serial or array, got '$GENERATION_MODE'"; exit 1 ;;
esac
# ARRAY_FOLLOWUP=1 marks the job the array-mode run chained after its arrays; it never
# re-submits tasks (a recurring failure would otherwise loop).
ARRAY_FOLLOWUP="${ARRAY_FOLLOWUP:-0}"
IMPORT_CPUS="${IMPORT_CPUS:-$SLURM_CPUS_PER_TASK}"
IMPORT_MEM_MB="${IMPORT_MEM_MB:-$SLURM_MEM_PER_NODE}"
GENERATE=1; IMPORT=1
if [ -n "$RESUME_JOB" ]; then
    GENERATE=0
    case "$RESUME_FROM" in generation|array) ;; *) IMPORT=0 ;; esac
fi
# The container installs torchcell from GitHub at this commit; the manifest records it.
# A resume takes the commit the CSVs were generated at from the old job's log.
git -C "$TORCHCELL_SRC" fetch origin main --quiet
if [ -n "$RESUME_JOB" ]; then
    PREV_OUT="$OUTDIR/${RESUME_JOB}_live_rebuild_kg.out"
    GEN_LOG="$OUTDIR/${RESUME_JOB}_generate.log"
    [ -f "$PREV_OUT" ] || { echo "job $RESUME_JOB left no $PREV_OUT"; exit 1; }
    if [ "$GENERATION_MODE" = serial ]; then
        [ -f "$GEN_LOG" ] || { echo "job $RESUME_JOB left no $GEN_LOG"; exit 1; }
    fi
    BUILD_COMMIT="${BUILD_COMMIT:-$(grep -oE '; commit [0-9a-f]{40}' "$PREV_OUT" | head -1 | grep -oE '[0-9a-f]{40}')}"
    # `|| true`: under set -e + pipefail a log with no organisms line (a job that predates
    # KG_ORGANISMS) would otherwise end the job here without the message below
    PREV_ORGANISMS=$(grep -oE '; organisms [a-z,]+' "$PREV_OUT" | head -1 | cut -d' ' -f3 || true)
    [ "$PREV_ORGANISMS" = "$KG_ORGANISMS" ] \
        || { echo "job $RESUME_JOB generated organisms '${PREV_ORGANISMS:-(not logged)}'; resubmit the resume with KG_ORGANISMS=${PREV_ORGANISMS:-<that set>}, not $KG_ORGANISMS"; exit 1; }
else
    BUILD_COMMIT="${BUILD_COMMIT:-$(git -C "$TORCHCELL_SRC" rev-parse origin/main)}"
fi
PIP_REF="git+https://github.com/Mjvolk3/torchcell.git@$BUILD_COMMIT"
# Pairing gate: a knowledge-graph release is served only as the pair of a package
# release, so the commit the store is built from must carry a package tag (vX.Y.Z,
# the bump commit semantic-release pushes for a REL / DB / PATCH / API commit), and the
# fingerprint checkout must sit on that same commit so `stamp` records the tag. A
# build from an untagged main would leave every published package incompatible with
# the store (the October 2026 builds did), and `releases retag` is the only repair.
PACKAGE_TAG=$(git -C "$TORCHCELL_SRC" describe --tags --exact-match --match 'v[0-9]*' "$BUILD_COMMIT" 2>/dev/null) \
    || { echo "commit $BUILD_COMMIT carries no package tag: cut the release first (a REL, DB, PATCH or API commit on main), wait for semantic-release to push the vX.Y.Z bump commit, then build from it"; exit 1; }
[ "$(git -C "$TORCHCELL_SRC" rev-parse HEAD)" = "$BUILD_COMMIT" ] \
    || { echo "$TORCHCELL_SRC HEAD is not $BUILD_COMMIT ($PACKAGE_TAG); check out the tag so the stamp records it"; exit 1; }
echo "package tag $PACKAGE_TAG at $BUILD_COMMIT: the release will be paired with it"

echo "Running on $(hostname) at $(date); job $JOB${RESUME_JOB:+ (resuming job $RESUME_JOB after its $RESUME_FROM)}"
echo "config $KG_CONFIG; commit $BUILD_COMMIT; dev tree $DEV_DATA_ROOT; serve $SERVE_ROOT; next $NEXT_ROOT; include private $INCLUDE_PRIVATE; organisms $KG_ORGANISMS"
echo "generation mode $GENERATION_MODE${RESUME_JOB:+ (resume from $RESUME_FROM)}${ARRAY_FOLLOWUP/1/ (array follow-up)}"
echo "SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK SLURM_MEM_PER_NODE=$SLURM_MEM_PER_NODE"
cd "$TORCHCELL_SRC"
export PYTHONPATH="$TORCHCELL_SRC"

echo "---------------- 0 PREFLIGHT"
docker inspect "$NEO4J_CONTAINER" --format '{{.State.Status}}' | grep -qx running \
    || { echo "$NEO4J_CONTAINER is not running"; exit 1; }
BEFORE=$(docker exec "$NEO4J_CONTAINER" $CS -d "$NEO4J_DATABASE" --format plain "MATCH (n) RETURN count(n);" | tail -1)
BEFORE_DS=$(docker exec "$NEO4J_CONTAINER" $CS -d "$NEO4J_DATABASE" --format plain "MATCH (d:Dataset) RETURN count(d);" | tail -1)
echo "served now: $BEFORE nodes, $BEFORE_DS datasets"
NEXT_PARENT=$(dirname "$NEXT_ROOT"); NEXT_NAME=$(basename "$NEXT_ROOT")
if [ -n "$RESUME_JOB" ]; then
    # The old job's preflight counted the mapped datasets against fresh dev stores and its
    # generation covered them all. Nothing here regenerates.
    N_DATASETS=$(grep -oE '^[0-9]+ mapped datasets' "$PREV_OUT" | head -1 | grep -oE '[0-9]+')
    if docker inspect "$GEN_CONTAINER" >/dev/null 2>&1; then echo "$GEN_CONTAINER still exists"; exit 1; fi
    if [ "$RESUME_FROM" = array ]; then
        # The array-mode job printed the directory the conductor created; the merge
        # below needs every task's DONE marker there. Nothing here regenerates unless
        # a task is missing and this is not a follow-up (stage 1 handles that).
        ARRAY_OUT=$(grep -oE '^ARRAY_OUT=.*' "$PREV_OUT" | tail -1 | cut -d= -f2-)
        [ -n "$ARRAY_OUT" ] || { echo "job $RESUME_JOB printed no ARRAY_OUT line: it was not an array-mode generation"; exit 1; }
        OUT_HOST="$SERVE_ROOT/biocypher-out/$ARRAY_OUT"
        for f in "$OUT_HOST/adapter_order.json" "$OUT_HOST/dev_fence.txt"; do
            [ -f "$f" ] || { echo "$f is absent; the conductor of job $RESUME_JOB did not run"; exit 1; }
        done
        if docker inspect "$BUILD_CONTAINER" >/dev/null 2>&1; then echo "$BUILD_CONTAINER already exists; remove it first"; exit 1; fi
        FREE_GB=$(df --output=avail -BG /db | tail -1 | tr -dc 0-9)
        [ "$FREE_GB" -ge "$MIN_FREE_GB" ] || { echo "/db has ${FREE_GB}G free, need ${MIN_FREE_GB}G"; exit 1; }
        if [ -e "$NEXT_ROOT/data" ] && [ -n "$(ls -A "$NEXT_ROOT/data")" ]; then
            echo "$NEXT_ROOT/data exists and is not empty; move it aside first"; exit 1
        fi
        DEV_FENCE="$OUT_HOST/dev_fence.txt"
        docker run --rm --entrypoint bash -v "$NEXT_PARENT":/parent "$IMAGE" -c \
            "mkdir -p /parent/$NEXT_NAME/data /parent/$NEXT_NAME/logs /parent/$NEXT_NAME/metrics \
             && chown -R 7474:7474 /parent/$NEXT_NAME && ls -la /parent/$NEXT_NAME"
        docker run --rm --entrypoint bash -v "$BUILD_ROOT/database":/bt "$IMAGE" -c \
            "chown -R $(id -u):$(id -g) /bt/conf /bt/biocypher && chmod -R u+rwX,go+rX /bt/conf /bt/biocypher"
        DATA_ROOT="$BUILD_ROOT" python -m torchcell.database.directory_setup --env-file "$DATABASE_ENV_FILE"
        cmp "$TORCHCELL_SRC/biocypher/config/torchcell_schema_config.yaml" \
            "$BUILD_ROOT/database/biocypher/config/torchcell_schema_config.yaml"
        echo "resuming after the array: $N_DATASETS mapped datasets, directories under $OUT_HOST/adapters, fence $DEV_FENCE"
        df -h /db /scratch
    elif [ "$IMPORT" = 0 ]; then
        # The import into $NEXT_ROOT/data completed and the build container was left
        # stopped. Nothing here reimports.
        grep -q '^IMPORT DONE in ' "$PREV_OUT" || { echo "job $RESUME_JOB did not finish its import"; exit 1; }
        STATE=$(docker inspect "$BUILD_CONTAINER" --format '{{.State.Status}}' 2>/dev/null || echo absent)
        # exited after the import, or still running from an earlier resume that failed later
        [ "$STATE" = "exited" ] || [ "$STATE" = "running" ] || { echo "$BUILD_CONTAINER is $STATE, expected exited or running"; exit 1; }
        [ -d "$NEXT_ROOT/data/databases/$NEO4J_DATABASE" ] || { echo "no imported $NEO4J_DATABASE under $NEXT_ROOT/data"; exit 1; }
        echo "resuming: $N_DATASETS mapped datasets, imported store $(du -sh "$NEXT_ROOT/data" | cut -f1)"
    else
        # The generator finished (its last line hands the CSVs over) and nothing was
        # imported: the build container must not exist and $NEXT_ROOT/data must be empty,
        # as for a fresh run. The dev fence is the generating job's, re-digested below.
        grep -q '^GENERATION_CALL=' "$GEN_LOG" \
            || { echo "job $RESUME_JOB left no GENERATION_CALL in $GEN_LOG: its generation did not finish (or its tail was not appended; see the header)"; exit 1; }
        if docker inspect "$BUILD_CONTAINER" >/dev/null 2>&1; then echo "$BUILD_CONTAINER already exists; remove it first"; exit 1; fi
        FREE_GB=$(df --output=avail -BG /db | tail -1 | tr -dc 0-9)
        [ "$FREE_GB" -ge "$MIN_FREE_GB" ] || { echo "/db has ${FREE_GB}G free, need ${MIN_FREE_GB}G"; exit 1; }
        if [ -e "$NEXT_ROOT/data" ] && [ -n "$(ls -A "$NEXT_ROOT/data")" ]; then
            echo "$NEXT_ROOT/data exists and is not empty; move it aside first"; exit 1
        fi
        DEV_FENCE_PY="$OUTDIR/${RESUME_JOB}_dev_fence.py"
        DEV_FENCE="$OUTDIR/${RESUME_JOB}_dev_fence.txt"
        for f in "$DEV_FENCE_PY" "$DEV_FENCE"; do [ -f "$f" ] || { echo "job $RESUME_JOB left no $f"; exit 1; }; done
        docker run --rm --entrypoint bash -v "$NEXT_PARENT":/parent "$IMAGE" -c \
            "mkdir -p /parent/$NEXT_NAME/data /parent/$NEXT_NAME/logs /parent/$NEXT_NAME/metrics \
             && chown -R 7474:7474 /parent/$NEXT_NAME && ls -la /parent/$NEXT_NAME"
        docker run --rm --entrypoint bash -v "$BUILD_ROOT/database":/bt "$IMAGE" -c \
            "chown -R $(id -u):$(id -g) /bt/conf /bt/biocypher && chmod -R u+rwX,go+rX /bt/conf /bt/biocypher"
        DATA_ROOT="$BUILD_ROOT" python -m torchcell.database.directory_setup --env-file "$DATABASE_ENV_FILE"
        cmp "$TORCHCELL_SRC/biocypher/config/torchcell_schema_config.yaml" \
            "$BUILD_ROOT/database/biocypher/config/torchcell_schema_config.yaml"
        echo "resuming after generation: $N_DATASETS mapped datasets, CSVs named by $GEN_LOG, fence $DEV_FENCE"
        df -h /db /scratch
    fi
else
# The fingerprint repo must carry the same code the container installs.
if ! git -C "$TORCHCELL_SRC" diff --quiet "$BUILD_COMMIT" HEAD -- torchcell biocypher; then
    echo "$TORCHCELL_SRC HEAD differs from $BUILD_COMMIT under torchcell/ or biocypher/"; exit 1
fi
for c in "$BUILD_CONTAINER" "$GEN_CONTAINER"; do
    if docker inspect "$c" >/dev/null 2>&1; then echo "$c already exists; remove it first"; exit 1; fi
done
FREE_GB=$(df --output=avail -BG /db | tail -1 | tr -dc 0-9)
[ "$FREE_GB" -ge "$MIN_FREE_GB" ] || { echo "/db has ${FREE_GB}G free, need ${MIN_FREE_GB}G"; exit 1; }
if [ -e "$NEXT_ROOT/data" ] && [ -n "$(ls -A "$NEXT_ROOT/data")" ]; then
    echo "$NEXT_ROOT/data exists and is not empty; move it aside first"; exit 1
fi
# Every mapped dataset must have a dev LMDB whose build manifest is fresh under the
# schema the container will install: the store is built from exactly those records. The
# check is `mapped_store_status`, the same function `build_dataset_lmdb --list-stale`
# runs, so the list the owner rebuilds from and the list this job refuses on cannot
# disagree (issue #833). It ends in a bounded read of each store's first record: a
# record pickled under a class this commit's schema no longer DEFINES is in no closure,
# so the fingerprints compare equal and only the read catches it -- 5 bacterial stores
# were in that state on 2026-10-09 while --list-stale named 1. The walk is the map of
# KG_ORGANISMS, so a yeast-only build counts and checks the yeast stores alone
# (build_dataset_lmdb --list-stale --organisms <set> prints the same list).
N_DATASETS=$(python - "$DEV_DATA_ROOT" "$INCLUDE_PRIVATE" "$KG_ORGANISMS" <<'EOF'
import sys
from torchcell.data.experiment_dataset import parse_organisms
from torchcell.database.build_dataset_lmdb import mapped_store_status
statuses = mapped_store_status(sys.argv[1], include_private=sys.argv[2] == "1", organisms=parse_organisms(sys.argv[3]))
bad = [s.describe() for s in statuses if s.needs_rebuild]
if bad:
    print("\n".join(bad), file=sys.stderr); sys.exit(1)
# The lane check (ontology unification guards, 2026-10-15): every fresh manifest carries
# perturbation_leaves, the leaf classes measured over every record at build time, and
# each leaf must be emitted by a node lane the dataset's adapter conf enables wherever
# the conf enables the edge that walks it. A leaf under no lane (Carruthers
# GeneAdditionPerturbation under a bacterial-only conf) would be a dangling
# `perturbation member of` row the import silently drops; refusing here costs seconds.
from torchcell.database.build_dataset_lmdb import mapped_store_lanes
lanes = mapped_store_lanes(sys.argv[1], include_private=sys.argv[2] == "1", organisms=parse_organisms(sys.argv[3]))
unserved = [s.describe() for s in lanes if s.blocks]
if unserved:
    print("\n".join(unserved), file=sys.stderr); sys.exit(1)
print(len(statuses))
EOF
)
echo "$N_DATASETS mapped datasets ($KG_ORGANISMS), every dev store fresh and every perturbation leaf served by its conf's lanes"
# The freshness check above is point-in-time, and generation then reads the dev tree for
# hours. Nothing stops a HOST process from rebuilding a store underneath it: the
# container's mounts are read-only, but `build_dataset_lmdb --retire-existing` renames
# processed/ and preprocess/ aside as this same user. Job 3547 died that way at its first
# bacterial adapter -- gene_interaction_babu2014 was rebuilt at 2026-10-09 04:35 CDT, two
# hours into the run and at a commit (a1d731d0, dirty) that is neither main nor the build
# tag, so the records the adapter read carried a GenePerturbationType member the installed
# v1.7.2 could not deserialize. The quiet version of that failure is worse: a store whose
# records still deserialize gets mixed into the graph under a provenance the manifest will
# not record. So fence the tree: digest every mapped dataset's build manifest now, and
# re-digest after generation, before anything is imported.
DEV_FENCE_PY="$OUTDIR/${JOB}_dev_fence.py"
cat > "$DEV_FENCE_PY" <<'EOF'
import hashlib, os, sys
from torchcell.data.experiment_dataset import parse_organisms
from torchcell.knowledge_graphs.dataset_adapter_map import build_adapter_map
from torchcell.database.build_dataset_lmdb import dataset_default_root
dev = sys.argv[1]
for cls in sorted(build_adapter_map(include_private=sys.argv[2] == "1", organisms=parse_organisms(sys.argv[3])), key=lambda c: c.__name__):
    mp = os.path.join(dev, dataset_default_root(cls), "preprocess", "build_manifest.json")
    with open(mp, "rb") as handle:
        print(f"{cls.__name__} {hashlib.sha256(handle.read()).hexdigest()}")
EOF
DEV_FENCE="$OUTDIR/${JOB}_dev_fence.txt"
python "$DEV_FENCE_PY" "$DEV_DATA_ROOT" "$INCLUDE_PRIVATE" "$KG_ORGANISMS" > "$DEV_FENCE"
echo "dev fence: $(wc -l < "$DEV_FENCE") build manifests digested into $DEV_FENCE"
docker run --rm --entrypoint bash -v "$NEXT_PARENT":/parent "$IMAGE" -c \
    "mkdir -p /parent/$NEXT_NAME/data /parent/$NEXT_NAME/logs /parent/$NEXT_NAME/metrics \
     && chown -R 7474:7474 /parent/$NEXT_NAME && ls -la /parent/$NEXT_NAME"
docker run --rm --entrypoint bash -v "$SERVE_ROOT":/serve "$IMAGE" -c \
    'mkdir -p /serve/biocypher-out && chown 7474:7474 /serve/biocypher-out && chmod 755 /serve/biocypher-out'
# build-tree biocypher config + container env come from the checkout (schema must match).
# The previous rebuild's container chowned $BUILD_ROOT/database/conf and biocypher to
# 7474 and chmodded them 700 (its entrypoint does that to everything under NEO4J_HOME),
# and directory_setup rewrites both as this user, so hand them back first.
docker run --rm --entrypoint bash -v "$BUILD_ROOT/database":/bt "$IMAGE" -c \
    "chown -R $(id -u):$(id -g) /bt/conf /bt/biocypher && chmod -R u+rwX,go+rX /bt/conf /bt/biocypher"
DATA_ROOT="$BUILD_ROOT" python -m torchcell.database.directory_setup --env-file "$DATABASE_ENV_FILE"
cmp "$TORCHCELL_SRC/biocypher/config/torchcell_schema_config.yaml" \
    "$BUILD_ROOT/database/biocypher/config/torchcell_schema_config.yaml"
df -h /db /scratch
fi

if [ "$GENERATION_MODE" = array ] && { [ "$GENERATE" = 1 ] || [ "$RESUME_FROM" = array ]; }; then
echo "---------------- 1 GENERATE as an array (one task per adapter) $(date)"
ARRAY_SCRIPT="$TORCHCELL_SRC/database/slurm/scripts/gilahyper_generate_adapter_array-slurm_docker.slurm"
CONDUCTOR="$TORCHCELL_SRC/database/scripts/kg_generation_array_conductor.sh"
if [ "$GENERATE" = 1 ]; then
    ARRAY_OUT=$(date +%Y-%m-%d_%H-%M-%S)
    OUT_HOST="$SERVE_ROOT/biocypher-out/$ARRAY_OUT"
    CONDUCTOR_LOG="$OUTDIR/${JOB}_conductor.log"
    OUT_NAME="$ARRAY_OUT" TORCHCELL_SRC="$TORCHCELL_SRC" BUILD_COMMIT="$BUILD_COMMIT" \
        DEV_DATA_ROOT="$DEV_DATA_ROOT" BUILD_ROOT="$BUILD_ROOT" SERVE_ROOT="$SERVE_ROOT" IMAGE="$IMAGE" \
        INCLUDE_PRIVATE="$INCLUDE_PRIVATE" KG_ORGANISMS="$KG_ORGANISMS" KG_CONFIG="$KG_CONFIG" \
        bash "$CONDUCTOR" 2>&1 | tee "$CONDUCTOR_LOG"
    # The conductor's fence is the one this build is held to; the preflight's own digest
    # must agree with it (both read the same tree minutes apart).
    diff "$DEV_FENCE" "$OUT_HOST/dev_fence.txt" || { echo "the conductor's fence differs from the preflight's; a store changed between them"; exit 1; }
    DEV_FENCE="$OUT_HOST/dev_fence.txt"
    MISSING_IDS=""
else
    # A resume: the tasks without a marker are re-run (by hand), or refused (follow-up).
    MISSING_IDS=$(python - "$OUT_HOST" <<'EOF'
import sys
from pathlib import Path
from torchcell.knowledge_graphs.generation_array import compress_ids, missing_slots, read_order
out = Path(sys.argv[1])
missing = missing_slots(out, read_order(out / "adapter_order.json"))
print(compress_ids(s.index for s in missing) if missing else "")
EOF
)
fi
if [ "$GENERATE" = 1 ] || [ -n "$MISSING_IDS" ]; then
    if [ "$GENERATE" = 0 ]; then
        echo "tasks without DONE: $MISSING_IDS"
        if [ "$ARRAY_FOLLOWUP" = 1 ]; then
            echo "this follow-up never re-runs a task. Fix the cause, then re-run them and chain a new follow-up with:"
            echo "  sbatch --export=ALL,RESUME_JOB=$RESUME_JOB,RESUME_FROM=array,TORCHCELL_SRC=$TORCHCELL_SRC,BUILD_COMMIT=$BUILD_COMMIT $0"
            echo "  (KG_ORGANISMS=$KG_ORGANISMS INCLUDE_PRIVATE=$INCLUDE_PRIVATE in the environment)"
            exit 1
        fi
        CONDUCTOR_LOG="$OUTDIR/${JOB}_conductor.log"
        OUT_NAME="$ARRAY_OUT" INDICES="$MISSING_IDS" TORCHCELL_SRC="$TORCHCELL_SRC" BUILD_COMMIT="$BUILD_COMMIT" \
            DEV_DATA_ROOT="$DEV_DATA_ROOT" BUILD_ROOT="$BUILD_ROOT" SERVE_ROOT="$SERVE_ROOT" IMAGE="$IMAGE" \
            INCLUDE_PRIVATE="$INCLUDE_PRIVATE" KG_ORGANISMS="$KG_ORGANISMS" KG_CONFIG="$KG_CONFIG" \
            bash "$CONDUCTOR" 2>&1 | tee "$CONDUCTOR_LOG"
    fi
    ARRAY_JOBS=$(grep -oE '^JOB_IDS=.*' "$CONDUCTOR_LOG" | tail -1 | cut -d= -f2-)
    [ -n "$ARRAY_JOBS" ] || { echo "the conductor printed no JOB_IDS"; exit 1; }
    echo "ARRAY_OUT=$ARRAY_OUT"
    echo "ARRAY_JOBS=$ARRAY_JOBS"
    # The follow-up: this script again, after every array, resuming at the merge. The
    # array-mode job that generated is the one a resume names (its log carries ARRAY_OUT),
    # so a follow-up of a follow-up still points at the generating job.
    GENERATING_JOB="${RESUME_JOB:-$JOB}"
    # IMPORT_CPUS / IMPORT_MEM_MB are not passed on: the follow-up sizes its import
    # container from its own reservation (the header's), whatever this job was given.
    FOLLOWUP=$(KG_ORGANISMS="$KG_ORGANISMS" INCLUDE_PRIVATE="$INCLUDE_PRIVATE" KG_CONFIG="$KG_CONFIG" \
        sbatch --parsable --dependency="afterany:${ARRAY_JOBS//,/:}" \
        --export=ALL,RESUME_JOB="$GENERATING_JOB",RESUME_FROM=array,ARRAY_FOLLOWUP=1,TORCHCELL_SRC="$TORCHCELL_SRC",BUILD_COMMIT="$BUILD_COMMIT" \
        "$TORCHCELL_SRC/database/slurm/scripts/gilahyper_live_rebuild-slurm_docker.slurm")
    echo "FOLLOWUP_JOB=$FOLLOWUP (after arrays $ARRAY_JOBS: merge, endpoint gate, import, validate, swap, record)"
    echo "array submitted at $(date); this job ends here"
    exit 0
fi
echo "every adapter directory carries DONE; merging $(date)"
MERGE_LOG="$OUTDIR/${JOB}_merge.log"
python -m torchcell.knowledge_graphs.generation_array merge --out-dir "$OUT_HOST" --workers "$SLURM_CPUS_PER_TASK" \
    --expect "$N_DATASETS" 2>&1 | tee "$MERGE_LOG"
grep -q '^MERGED_CALL=' "$MERGE_LOG" || { echo "the merge wrote no import call"; exit 1; }
MERGE_REPORT="$OUT_HOST/merge_report.json"
[ -f "$MERGE_REPORT" ] || { echo "no $MERGE_REPORT"; exit 1; }
# The CSV endpoint gate runs in the import branch below for every generation mode
# (csv_endpoint_gate discovers adapters/*/ as one output), after the fence re-digest.
GEN_LOG=""
elif [ "$GENERATE" = 1 ]; then
echo "---------------- 1 GENERATE (ephemeral container; dev tree read-only) $(date)"
# The dev tree's data/torchcell shadows the build tree's copy read-only (loaders open
# LMDB readonly, lock=False); yeast genome/graph caches (data/sgd, go, string, tflink)
# come from the build tree as in every previous build. The bacterial genome caches
# (data/ecoli/<strain>/genome, data/pputida/kt2440/genome) are read from the dev tree
# read-only: the bacterial adapters build their reference genome at generation time
# with overwrite=False, and job 3546 (the first build with bacterial datasets) died in
# the generation container on GenomeRootNotFoundError because the build tree has no
# such cache. The container runs as root so pip can install into the image; the CSVs
# are handed to uid 7474, which runs the import.
GEN_LOG="$OUTDIR/${JOB}_generate.log"
docker run --rm --entrypoint bash --name "$GEN_CONTAINER" \
    --cpus="$SLURM_CPUS_PER_TASK" -e SLURM_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK" --memory="${mem_mb}m" \
    -e NEO4J_ACCEPT_LICENSE_AGREEMENT=yes \
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
        python -m $KG_MODULE --config-name $KG_CONFIG $PRIVATE_FLAG --organisms $KG_ORGANISMS
        CALL=\$(cat biocypher_file_name.txt)
        OUT_DIR=\$(dirname \$CALL)
        chown -R 7474:7474 /var/lib/neo4j/\$OUT_DIR && chmod -R a+rX /var/lib/neo4j/\$OUT_DIR
        echo GENERATION_CALL=\$CALL" 2>&1 | tee "$GEN_LOG"
echo "CSV generation completed at $(date)"
fi
if [ "$GENERATION_MODE" = serial ]; then
    N_ADAPTERS=$(grep -oE 'across [0-9]+ adapters' "$GEN_LOG" | tail -1 | grep -oE '[0-9]+')
    [ "$N_ADAPTERS" = "$N_DATASETS" ] || { echo "generation covered '$N_ADAPTERS' adapters, expected $N_DATASETS"; exit 1; }
fi
# The fence: the dev tree must be byte-identical to what the preflight saw, or the CSVs
# mix two schema versions. A resume after the import regenerates and reimports nothing,
# so its fence was already honored by the job that generated; a resume after the
# generation re-digests against the generating job's fence before it imports.
if [ "$IMPORT" = 1 ]; then
    DEV_FENCE_AFTER="$OUTDIR/${JOB}_dev_fence_after.txt"
    if [ "$GENERATION_MODE" = array ]; then
        python -m torchcell.knowledge_graphs.generation_array fence --dev-data-root "$DEV_DATA_ROOT" \
            $PRIVATE_FLAG --organisms "$KG_ORGANISMS" > "$DEV_FENCE_AFTER"
    else
        python "$DEV_FENCE_PY" "$DEV_DATA_ROOT" "$INCLUDE_PRIVATE" "$KG_ORGANISMS" > "$DEV_FENCE_AFTER"
    fi
    if diff "$DEV_FENCE" "$DEV_FENCE_AFTER"; then
        echo "dev fence intact: all $N_DATASETS build manifests unchanged across generation"
    else
        echo "a dev store was rebuilt DURING generation (diff above); these CSVs mix schema versions"
        echo "rebuild the stale stores first, then resubmit with the tree quiet"
        exit 1
    fi
fi
if [ "$GENERATION_MODE" = array ]; then
    CALL_REL="biocypher-out/$ARRAY_OUT/neo4j-admin-import-call.sh"
else
    CALL_REL=$(grep -oE '^GENERATION_CALL=.*' "$GEN_LOG" | tail -1 | cut -d= -f2-)
fi
[ -n "$CALL_REL" ] || { echo "no import call reported"; exit 1; }
OUT_TS=$(basename "$(dirname "$CALL_REL")")
OUT_HOST="$SERVE_ROOT/biocypher-out/$OUT_TS"
OUT_CONT="/var/lib/neo4j/biocypher-out/$OUT_TS"
echo "CSVs: $OUT_HOST ($CALL_REL)"
df -h /db

if [ "$IMPORT" = 1 ]; then
# The CSV endpoint gate, a hard stop before the import (ontology unification guards,
# 2026-10-15). Every edge row's :START_ID and :END_ID must be the :ID of a node row;
# neo4j-admin import runs with skip_bad_relationships: true (biocypher/config/*.yaml)
# and DROPS such a row, reporting it only in an import.report the validate stage below
# prints twenty lines of. Flipping that flag alone would move the failure into the
# import after the whole generation (review 3 finding 8), so the flag stays as it is
# and this check is the gate: minutes over the CSVs, and it names the edge label, the
# side and the dataset of every dangling row. The expected-counts block after the
# import (per-dataset ExperimentMemberOf counts) is the live-store counterpart.
ENDPOINT_REPORT="$OUTDIR/${JOB}_csv_endpoint_gate.json"
if python -m torchcell.knowledge_graphs.csv_endpoint_gate "$OUT_HOST" --report "$ENDPOINT_REPORT" 2>&1 | tee "$OUTDIR/${JOB}_csv_endpoint_gate.txt"; then
    echo "CSV endpoint gate passed: every edge endpoint is a node ($ENDPOINT_REPORT)"
else
    echo "CSV endpoint gate FAILED: an edge endpoint names no node (see $ENDPOINT_REPORT); the fix is in the data, regenerate after it"
    exit 1
fi
echo "---------------- 2 IMPORT into $NEXT_ROOT ($BUILD_CONTAINER, no host ports) $(date)"
IMPORT_T0=$(date +%s)
docker run \
    --cpus="$IMPORT_CPUS" --memory="${IMPORT_MEM_MB}m" \
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
# The marker a RESUME_FROM=import reads (the header's "RESUME after a failure past the
# import"); it was documented but never written before job 4901.
echo "IMPORT DONE in $(( $(date +%s) - IMPORT_T0 )) s at $(date)"
else
echo "---------------- 2 IMPORT done by job $RESUME_JOB; restarting $BUILD_CONTAINER $(date)"
fi
# The index-stats file the read-only container needs is written on a stop/start + query.
# Restart the CONTAINER: the image entrypoint runs neo4j as PID 1, so `neo4j stop` inside
# it ends the container, and with --restart=no it stays down (job 2032 died on the
# `neo4j start` that followed). docker stop on an already-exited container is a no-op.
docker stop "$BUILD_CONTAINER"
docker start "$BUILD_CONTAINER"
sleep 60
for i in $(seq 1 60); do
    STATUS=$(docker exec "$BUILD_CONTAINER" $CS -d system --format plain \
        "SHOW DATABASE $NEO4J_DATABASE YIELD currentStatus RETURN currentStatus;" 2>/dev/null | tail -1 | tr -d '"')
    [ "$STATUS" = "online" ] && break
    echo "  $NEO4J_DATABASE is ${STATUS:-starting} after restart ($((i*10))s)"; sleep 10
done
[ "$STATUS" = "online" ] || { echo "$NEO4J_DATABASE did not come back online in $BUILD_CONTAINER; see docker logs $BUILD_CONTAINER"; exit 1; }
NEW_COUNT=$(docker exec "$BUILD_CONTAINER" $CS -d "$NEO4J_DATABASE" --format plain "MATCH (n) RETURN count(n);" | tail -1)
echo "imported store: $NEW_COUNT nodes (served store: $BEFORE)"
# Make torchcell the default database of the new store, so a Browser or driver session
# that names no database lands on it and nobody types `:use torchcell`. The setting
# lives in the system database, which travels with the data directory through the swap.
# The procedure needs the old default (the empty `neo4j`) stopped; it is started again.
# Guarded so a resume of a store that already has it does not stop a default in use.
DEFAULT_DB=$(docker exec "$BUILD_CONTAINER" $CS -d system --format plain "SHOW DEFAULT DATABASE YIELD name RETURN name;" | tail -1 | tr -d '"')
if [ "$DEFAULT_DB" != "$NEO4J_DATABASE" ]; then
    docker exec "$BUILD_CONTAINER" $CS -d system \
        "STOP DATABASE $DEFAULT_DB; CALL dbms.setDefaultDatabase('$NEO4J_DATABASE'); START DATABASE $DEFAULT_DB;"
fi
DEFAULT_DB=$(docker exec "$BUILD_CONTAINER" $CS -d system --format plain "SHOW DEFAULT DATABASE YIELD name RETURN name;" | tail -1 | tr -d '"')
[ "$DEFAULT_DB" = "$NEO4J_DATABASE" ] || { echo "default database is $DEFAULT_DB, expected $NEO4J_DATABASE"; exit 1; }
echo "default database: $DEFAULT_DB"

echo "---------------- 3 VALIDATE before touching the served store"
VALIDATION="$OUTDIR/${JOB}_validation.txt"
NEW_DS=$(docker exec "$BUILD_CONTAINER" $CS -d "$NEO4J_DATABASE" --format plain "MATCH (d:Dataset) RETURN count(d);" | tail -1)
echo "Dataset nodes: $NEW_DS (expected $N_DATASETS)" | tee "$VALIDATION"
[ "$NEW_DS" = "$N_DATASETS" ] || { echo "dataset count mismatch"; exit 1; }
# ExperimentMemberOf rows per Dataset in the CSVs are what the store must hold. The CSVs
# are read on the HOST as the dev user, and every start of an entrypoint container
# (the build container above, the serving container after the swap) runs
# `chmod -R 700` on each top-level directory under NEO4J_HOME, the mounted
# biocypher-out included (job 2311 failed here with Permission denied). Reopen it first.
docker run --rm --entrypoint bash -v "$SERVE_ROOT/biocypher-out":/out "$IMAGE" -c \
    "chmod 755 /out && chmod -R a+rX /out/$OUT_TS"
EXPECTED="$OUTDIR/${JOB}_expected.txt"
python - "$OUT_HOST" > "$EXPECTED" <<'EOF'
import csv, sys
from collections import Counter
from pathlib import Path
from torchcell.knowledge_graphs.incremental_import import discover_csv_groups
csv.field_size_limit(sys.maxsize)
out_dir = Path(sys.argv[1])
# serial: the CSVs sit in out_dir; array: one directory per adapter under adapters/
adapters_dir = out_dir / "adapters"
csv_dirs = sorted(p for p in adapters_dir.iterdir() if p.is_dir() and ".incomplete." not in p.name) if adapters_dir.is_dir() else [out_dir]
counts: Counter[str] = Counter()
for csv_dir in csv_dirs:
    group = next((g for g in discover_csv_groups(csv_dir) if g.label == "ExperimentMemberOf"), None)
    if group is None:
        continue
    end = group.columns.index(":END_ID")
    for part in group.part_paths:
        with open(part, encoding="utf-8", newline="") as handle:
            for row in csv.reader(handle, delimiter="\t", quotechar="'", strict=True):
                counts[row[end].strip("'")] += 1
for name, n in sorted(counts.items()):
    print(f"{name}={n}")
EOF
# Neo4j 5.26 rejects an aggregation expression that also carries a grouping key
# (d.id + '=' + toString(count(e)) failed job 2325 as "implicit grouping"), so the id and
# the count are returned as two columns and joined here. Both files are sorted in the C
# locale, the order Python's sorted() produced for the CSV side.
docker exec "$BUILD_CONTAINER" $CS -d "$NEO4J_DATABASE" --format plain \
    "MATCH (d:Dataset)<-[:ExperimentMemberOf]-(e:Experiment) RETURN d.id AS id, count(e) AS n ORDER BY id;" \
    | tail -n +2 | tr -d '" ' | tr ',' '=' | LC_ALL=C sort > "$OUTDIR/${JOB}_live.txt"
LC_ALL=C sort -o "$EXPECTED" "$EXPECTED"
if diff "$EXPECTED" "$OUTDIR/${JOB}_live.txt" | tee -a "$VALIDATION"; then
    echo "per-dataset experiment counts: live == CSV for all $N_DATASETS datasets" | tee -a "$VALIDATION"
else
    echo "per-dataset experiment counts differ between the CSVs and the store"; exit 1
fi
echo "import.report (first 20 lines):" | tee -a "$VALIDATION"
docker exec "$BUILD_CONTAINER" bash -c "f=\$(ls $OUT_CONT/import.report /var/lib/neo4j/import.report /var/lib/neo4j/data/import.report 2>/dev/null | head -1); [ -n \"\$f\" ] && head -20 \"\$f\" || echo '(no import.report found)'" | tee -a "$VALIDATION"
echo "total nodes: $NEW_COUNT (served before: $BEFORE)" | tee -a "$VALIDATION"
# Per-dataset content hashes (sha256 of the sorted experiment ids), the byte identity a
# release records per dataset; streamed through cypher-shell from the validated store.
HASHES="$OUTDIR/${JOB}_content_hashes.json"
bash "$TORCHCELL_SRC/database/scripts/kg_content_hashes.sh" "$BUILD_CONTAINER" "$NEO4J_DATABASE" "$HASHES"
echo "content hashes: $HASHES" | tee -a "$VALIDATION"

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
    -v "$RELEASE_ROOT":/kg-releases \
    -e NEO4J_AUTH=neo4j/torchcell \
    -e NEO4J_server_databases_default__to__read__only=true \
    "$IMAGE"
# Under set -e + pipefail an assignment from a failing pipeline exits the job, and
# cypher-shell fails until neo4j accepts connections, so the first poll ran too early and
# killed job 2331 AFTER the swap (serving came up fine on its own). Wait first, and let a
# not-yet-listening poll read as "starting".
sleep 60
for i in $(seq 1 120); do
    STATUS=$(docker exec "$NEO4J_CONTAINER" $CS -d system --format plain \
        "SHOW DATABASE $NEO4J_DATABASE YIELD currentStatus RETURN currentStatus;" 2>/dev/null | tail -1 | tr -d '"' || true)
    [ "$STATUS" = "online" ] && break
    echo "  serving $NEO4J_DATABASE is ${STATUS:-starting} ($((60 + i*10))s)"; sleep 10
done
[ "$STATUS" = "online" ] || { echo "served $NEO4J_DATABASE did not come online; see docker logs $NEO4J_CONTAINER"; exit 1; }
SERVED=$(docker exec "$NEO4J_CONTAINER" $CS -d "$NEO4J_DATABASE" --format plain "MATCH (n) RETURN count(n);" | tail -1)
echo "SERVING $SERVED nodes (validated $NEW_COUNT) at $(date)"
[ "$SERVED" = "$NEW_COUNT" ] || { echo "served count differs from the validated store"; exit 1; }
docker exec "$NEO4J_CONTAINER" $CS -d system "SHOW DATABASES YIELD name, default, home, currentStatus, access;"
SERVED_DEFAULT=$(docker exec "$NEO4J_CONTAINER" $CS -d system --format plain "SHOW DEFAULT DATABASE YIELD name RETURN name;" | tail -1 | tr -d '"')
[ "$SERVED_DEFAULT" = "$NEO4J_DATABASE" ] || { echo "served default database is $SERVED_DEFAULT, expected $NEO4J_DATABASE"; exit 1; }
# The served Browser page must carry the styling seed (the image is the one that has it).
curl -sf http://localhost:7474/browser/ | grep -q 'torchcell-seed.js' \
    || { echo "served Browser page has no torchcell-seed.js tag: is $IMAGE the seeded image?"; exit 1; }

echo "---------------- 5 RECORD + ARCHIVE"
if [ -f "$KG_MANIFEST" ]; then
    mv "$KG_MANIFEST" "$KG_MANIFEST.superseded.$SWAP_TS"
    echo "previous manifest kept at $KG_MANIFEST.superseded.$SWAP_TS"
fi
BUILT_AT=$(date -Iseconds)
python -m torchcell.knowledge_graphs.kg_manifest --manifest "$KG_MANIFEST" --repo "$TORCHCELL_SRC" \
    bootstrap --commit "$BUILD_COMMIT" --biocypher-out "$OUT_TS" --built-at "$BUILT_AT" \
    --biocypher-version 0.15.2 --database "$NEO4J_DATABASE" --neo4j-uri bolt://localhost:7687 \
    ${MERGE_REPORT:+--merge-report "$MERGE_REPORT"} \
    --organisms "$KG_ORGANISMS"
python -m torchcell.knowledge_graphs.kg_manifest --manifest "$KG_MANIFEST" --repo "$TORCHCELL_SRC" show
# Release: a full build bumps the MAJOR version of the store it replaces (the superseded
# manifest's version; none for the first stamped store), takes <build date>-<commit> as
# its release id, records each dataset's content hash, and writes the KgRelease node the
# store carries from now on (the served database is opened for writing for that one
# statement through the dynamic server.databases.writable setting, as the increment
# runner does). The latest and pinned aliases both point at the one physical database
# a GilaHyper DBMS holds after a swap; a DBMS that keeps several releases retargets them.
PREV_VERSION=""
if [ -f "$KG_MANIFEST.superseded.$SWAP_TS" ]; then
    PREV_VERSION=$(python -c 'import json,sys; print(json.load(open(sys.argv[1])).get("version") or "")' "$KG_MANIFEST.superseded.$SWAP_TS")
fi
# The pointer set: every dataset whose closure names ArtifactRef is walked once in the
# dev tree, so the KgRelease node carries the files `make ops` checks against tc-data.
python -m torchcell.knowledge_graphs.kg_manifest --manifest "$KG_MANIFEST" --repo "$TORCHCELL_SRC" \
    artifact-refs --data-root "$DEV_DATA_ROOT"
python -m torchcell.knowledge_graphs.releases stamp --manifest "$KG_MANIFEST" --kind full \
    --built-at "$BUILT_AT" --hashes "$HASHES" ${PREV_VERSION:+--previous-version "$PREV_VERSION"}
docker exec "$NEO4J_CONTAINER" $CS -d system "CALL dbms.setConfigValue('server.databases.writable', '$NEO4J_DATABASE');"
python -m torchcell.knowledge_graphs.releases write-node --manifest "$KG_MANIFEST" --database "$NEO4J_DATABASE" \
    --built-at "$BUILT_AT" --n-nodes "$SERVED"
# Release snapshot (plan.data-release-program.2026.09.29, Decision 1): the committed
# database/releases/<release>.json (+ .closures.json) that scripts/kg_compat_page.py
# turns into the docs compatibility page. Written into the checkout; commit it as
# `DB(kg): <release> snapshot ...` (DB is a patch bump; `RELEASE(kg)` is not an allowed
# tag and bumps nothing), then regenerate the page with `python scripts/kg_compat_page.py`.
python -m torchcell.knowledge_graphs.releases snapshot --manifest "$KG_MANIFEST" \
    --built-at "$BUILT_AT" --n-nodes "$SERVED" --repo-root "$TORCHCELL_SRC"
docker exec "$NEO4J_CONTAINER" $CS -d system "CALL dbms.setConfigValue('server.databases.writable', '');"
docker exec "$NEO4J_CONTAINER" $CS -d system \
    "CREATE ALIAS latest IF NOT EXISTS FOR DATABASE $NEO4J_DATABASE; CREATE ALIAS pinned IF NOT EXISTS FOR DATABASE $NEO4J_DATABASE;"
python -m torchcell.knowledge_graphs.releases --repo "$TORCHCELL_SRC" status --label gilahyper
docker run --rm --entrypoint bash --user 7474:7474 -v "$SERVE_ROOT/biocypher-out":/src:ro -v "$ARCHIVE_ROOT":/dst "$IMAGE" \
    -c "cp -a /src/$OUT_TS /dst/ && ls -ld /dst/$OUT_TS"
df -h /db /scratch /bulk
echo "DONE at $(date): $SERVED nodes, $N_DATASETS datasets ($KG_ORGANISMS) served from $SERVE_ROOT/data; CSVs $OUT_HOST (archived to $ARCHIVE_ROOT/$OUT_TS); validation $VALIDATION"
