#!/bin/bash
#SBATCH --job-name=tcdb002-bench
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=192G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/projects/torchcell-scratch/tcdb-002-bench/slurm/%j_%x.out
#SBATCH --error=/scratch/projects/torchcell-scratch/tcdb-002-bench/slurm/%j_%x.out
#SBATCH --no-requeue
#
# experiments/tcdb-002-build-speed/scripts/gh_bench_generate.slurm
# [[experiments.tcdb-002-build-speed.scripts.gh_bench_generate]]
#
# One ARM of the build-speed benchmark: CSV generation only (the 28 h of a 28.5 h full
# build), in the same container and with the same mounts as the production rebuild
# (database/slurm/scripts/gilahyper_live_rebuild-slurm_docker.slurm, step 1 GENERATE),
# over the fixed ladder config. No import. The container's cgroup is the box the arm
# is scored in: --cpus and --memory come from this job's allocation.
#
# Submit through submit_bench.sh, never directly: the wrapper builds the torchcell
# wheel from the worktree AT SUBMIT TIME, so a queued arm runs the commit it was
# submitted from even if the worktree moves on (sbatch copies the script, not the
# source tree). Inputs (exported by the wrapper):
#   WHEEL_DIR   host dir holding torchcell-*.whl
#   ROUND ARM   round id (r0, r1, ...) and arm name (baseline, inproc-small, ...)
#   COMMIT      short sha the wheel was built from (a W&B tag)
#   KG_CONFIG   hydra config name (default kg_bench_ladder)
#   KG_OVERRIDES  space-separated hydra overrides that define the arm's deviation from
#               the config (e.g. "adapters.inprocess_max_records=25000"); empty = none
#   KEEP_CSV    1 keeps the generated CSVs; default 0 removes them after the row counts
#               and byte sizes are recorded (a ladder arm writes tens of GB)
set -euo pipefail

JOB="${SLURM_JOB_ID}"
BENCH_ROOT="${BENCH_ROOT:-/scratch/projects/torchcell-scratch/tcdb-002-bench}"
DEV_DATA_ROOT="${DEV_DATA_ROOT:-/scratch/projects/torchcell-scratch}"
BUILD_ROOT="${BUILD_ROOT:-/scratch/projects/torchcell}"
IMAGE="${IMAGE:-michaelvolk/tc-neo4j:5.26.28-browser.1}"
KG_MODULE="${KG_MODULE:-torchcell.knowledge_graphs.create_scerevisiae_kg_small}"
KG_CONFIG="${KG_CONFIG:-kg_bench_ladder}"
KG_OVERRIDES="${KG_OVERRIDES:-}"
KEEP_CSV="${KEEP_CSV:-0}"
: "${WHEEL_DIR:?submit through submit_bench.sh}" "${ROUND:?}" "${ARM:?}" "${COMMIT:?}"

RUN_DIR="$BENCH_ROOT/runs/${JOB}_${ROUND}_${ARM}"
mkdir -p "$RUN_DIR" "$BENCH_ROOT/biocypher-out"
WHEEL=$(ls "$WHEEL_DIR"/torchcell-*.whl | head -1)
[ -n "$WHEEL" ] || { echo "no wheel in $WHEEL_DIR"; exit 1; }
mem_mb=$((SLURM_MEM_PER_NODE))
echo "arm=$ARM round=$ROUND commit=$COMMIT job=$JOB cpus=$SLURM_CPUS_PER_TASK mem_mb=$mem_mb wheel=$(basename "$WHEEL") overrides='$KG_OVERRIDES'"
echo "started $(date)"
GEN_CONTAINER="tcdb002-gen-${JOB}"
GEN_LOG="$RUN_DIR/generate.log"

# Same mounts as the production generate step: build tree for genome/graph caches and
# the container env, dev tree datasets read-only, CSVs to the bench output dir.
docker run --rm --entrypoint bash --name "$GEN_CONTAINER" \
    --cpus="$SLURM_CPUS_PER_TASK" -e SLURM_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK" --memory="${mem_mb}m" \
    -e NEO4J_ACCEPT_LICENSE_AGREEMENT=yes \
    -e TCDB_TAGS="${ROUND},${ARM},commit-${COMMIT},cpus-${SLURM_CPUS_PER_TASK},mem-${mem_mb}m" \
    -e TCDB_JOB_TYPE=bench -e SLURM_JOB_ID="$JOB" \
    -v "$BENCH_ROOT/biocypher-out":/var/lib/neo4j/biocypher-out \
    -v "$BUILD_ROOT/database/data":/var/lib/neo4j/data \
    -v "$DEV_DATA_ROOT/data/torchcell":/var/lib/neo4j/data/torchcell:ro \
    -v "$DEV_DATA_ROOT/torchcell-genomes":/var/lib/neo4j/torchcell-genomes:ro \
    -v "$BUILD_ROOT/database/.env":/.env \
    -v "$BUILD_ROOT/database/biocypher":/var/lib/neo4j/biocypher \
    -v "$WHEEL_DIR":/mnt/torchcell-wheel:ro \
    "$IMAGE" -c "set -euo pipefail
        cd /var/lib/neo4j
        python -m pip install --no-cache-dir --force-reinstall --no-deps '/mnt/torchcell-wheel/$(basename "$WHEEL")' >/dev/null
        python -m pip install --no-cache-dir biocypher==0.15.2 >/dev/null
        python -m pip install --no-cache-dir git+https://github.com/oxpig/CaLM@main >/dev/null
        set -a; source /.env; set +a
        wandb login \$WANDB_API_KEY >/dev/null
        echo container_cpu_max=\$(cat /sys/fs/cgroup/cpu.max) memory_max=\$(cat /sys/fs/cgroup/memory.max)
        python -m $KG_MODULE --config-name $KG_CONFIG $KG_OVERRIDES
        CALL=\$(cat biocypher_file_name.txt)
        OUT_DIR=/var/lib/neo4j/\$(dirname \$CALL)
        echo GENERATION_OUT=\$OUT_DIR
        # Score inputs: bytes and rows per CSV, the telemetry, the phase table.
        (cd \$OUT_DIR && for f in *.csv; do printf '%s\t%s\t%s\n' \"\$f\" \$(stat -c %s \"\$f\") \$(wc -l < \"\$f\"); done) > \$OUT_DIR/csv_inventory.tsv
        du -sb \$OUT_DIR | cut -f1 > \$OUT_DIR/csv_total_bytes
        cp -r \$OUT_DIR/telemetry \$OUT_DIR/csv_inventory.tsv \$OUT_DIR/csv_total_bytes /var/lib/neo4j/biocypher-out/_run_${JOB}/ 2>/dev/null || { mkdir -p /var/lib/neo4j/biocypher-out/_run_${JOB}; cp -r \$OUT_DIR/telemetry \$OUT_DIR/csv_inventory.tsv \$OUT_DIR/csv_total_bytes /var/lib/neo4j/biocypher-out/_run_${JOB}/; }
        cp /var/lib/neo4j/biocypher_warnings.log /var/lib/neo4j/biocypher-out/_run_${JOB}/ || true
        if [ '$KEEP_CSV' != '1' ]; then rm -rf \$OUT_DIR; echo 'CSVs removed (KEEP_CSV=0)'; fi
        chown -R $(id -u):$(id -g) /var/lib/neo4j/biocypher-out/_run_${JOB}
        [ '$KEEP_CSV' = '1' ] && chown -R $(id -u):$(id -g) \$OUT_DIR || true" 2>&1 | tee "$GEN_LOG"

mv "$BENCH_ROOT/biocypher-out/_run_${JOB}"/* "$RUN_DIR/"
rmdir "$BENCH_ROOT/biocypher-out/_run_${JOB}"
sacct -j "$JOB" --format=JobID,ReqCPUS,AllocCPUS,ReqMem,Elapsed,MaxRSS,State -P > "$RUN_DIR/sacct.txt"
printf 'job\t%s\nround\t%s\narm\t%s\ncommit\t%s\ncpus\t%s\nmem_mb\t%s\nkg_config\t%s\noverrides\t%s\nwheel\t%s\n' \
    "$JOB" "$ROUND" "$ARM" "$COMMIT" "$SLURM_CPUS_PER_TASK" "$mem_mb" "$KG_CONFIG" "$KG_OVERRIDES" "$(basename "$WHEEL")" > "$RUN_DIR/arm.tsv"
echo "finished $(date); results in $RUN_DIR"
