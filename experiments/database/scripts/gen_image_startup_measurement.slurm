#!/bin/bash
#SBATCH --job-name=gen-image-measure
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/projects/torchcell/database/slurm/output/%j_%x.out
#SBATCH --error=/scratch/projects/torchcell/database/slurm/output/%j_%x.out
#SBATCH --no-requeue

# experiments/database/scripts/gen_image_startup_measurement.slurm
# [[experiments.database.scripts.gen_image_startup_measurement]]
#
# Measure what the prebuilt generation image (database/docker/Dockerfile.tc-neo4j-gen,
# torchcell.knowledge_graphs.generation_image) saves per array task: the wall time from
# `docker run` to the generator's first own log line, for ONE small adapter, in the
# install path (the base image plus the three pip installs the serial build runs) and
# in the image path, RUNS times each. Every container line is stamped on the host with
# the UTC epoch, and the markers are
#
#   T0_RUN_START         host, just before docker run
#   CONTAINER_SHELL_UP   the container's first command
#   INSTALL_DONE         install path: after the three pip installs
#   COMMIT_VERIFIED      image path: after the direct_url.json commit check
#   Generating adapter   the generator's first own line (generate_adapter.main)
#   DONE=                the generator's last line (the marker written)
#   T1_RUN_END           host, after the container exited
#
# experiments/database/scripts/gen_image_startup_measurement.py reads the job's .out
# and writes experiments/database/results/gen_image_startup_<job>.{json,md} with the
# per-run intervals, the image build times, and the projected array saving.
#
# Two images are built, both timed: MAIN_COMMIT (the deliverable, `tc-neo4j-gen:<sha12>`
# of main) and GEN_COMMIT, the commit whose generator runs (generate_adapter lives on
# lane C's branch until PR #1110 lands; the default is that branch's head). Startup is
# a property of the container, not of the adapter, so the generator commit only has to
# carry generate_adapter. The outputs go under $MEASURE_ROOT/<job>, never under
# /db/database. Sized for the small adapter alone (16 CPU, 64 G, 2 h); runs one
# container at a time.
#
#   sbatch --export=ALL,TORCHCELL_SRC=<wt>,MAIN_COMMIT=<sha>,GEN_COMMIT=<sha> \
#       experiments/database/scripts/gen_image_startup_measurement.slurm
#
# Variables: TORCHCELL_SRC (the checkout with the Dockerfile), MAIN_COMMIT, GEN_COMMIT,
# ADAPTER_INDEX (85, IsopentenolTiterFoo2014Adapter: 56 K of LMDB), RUNS (3),
# REBUILD (1: remove both tags first so the build is measured from nothing),
# KEEP_MAIN_IMAGE (1: the deliverable image stays for inspection; GEN_COMMIT's is
# always removed), DEV_DATA_ROOT, BUILD_ROOT, IMAGE as in the live rebuild.

source ~/miniconda3/etc/profile.d/conda.sh
conda activate torchcell
set -euo pipefail

TORCHCELL_SRC="${TORCHCELL_SRC:?set TORCHCELL_SRC=<checkout carrying the Dockerfile>}"
MAIN_COMMIT="${MAIN_COMMIT:?set MAIN_COMMIT=<full sha of main>}"
GEN_COMMIT="${GEN_COMMIT:?set GEN_COMMIT=<full sha carrying generate_adapter>}"
ADAPTER_INDEX="${ADAPTER_INDEX:-85}"
RUNS="${RUNS:-3}"
REBUILD="${REBUILD:-1}"
KEEP_MAIN_IMAGE="${KEEP_MAIN_IMAGE:-1}"
DEV_DATA_ROOT="${DEV_DATA_ROOT:-/scratch/projects/torchcell-scratch}"
BUILD_ROOT="${BUILD_ROOT:-/scratch/projects/torchcell}"
IMAGE="${IMAGE:-michaelvolk/tc-neo4j:5.26.28-browser.1}"
KG_CONFIG="${KG_CONFIG:-kg_uncapped}"
KG_ORGANISMS="${KG_ORGANISMS:-scerevisiae,ecoli,pputida}"
CPUS="${SLURM_CPUS_PER_TASK:-16}"
MEM_GB=$(( ${SLURM_MEM_PER_NODE:-65536} / 1024 ))
JOB="${SLURM_JOB_ID:-nojob}"
MEASURE_ROOT="${MEASURE_ROOT:-$BUILD_ROOT/database/gen-image-measure}/$JOB"
PIP_REF="git+https://github.com/Mjvolk3/torchcell.git@$GEN_COMMIT"
cd "$TORCHCELL_SRC"
export PYTHONPATH="$TORCHCELL_SRC"
GI="python -m torchcell.knowledge_graphs.generation_image"

stamp() { while IFS= read -r line; do printf '%s %s\n' "$(date -u +%s.%N)" "$line"; done; }
mark() { printf '%s %s\n' "$(date -u +%s.%N)" "$*"; }

echo "Running on $(hostname) at $(date); job $JOB; $CPUS cpus, ${MEM_GB}G"
echo "main commit $MAIN_COMMIT; generator commit $GEN_COMMIT; adapter index $ADAPTER_INDEX; runs $RUNS; out $MEASURE_ROOT"
mkdir -p "$MEASURE_ROOT"
docker ps -a --filter name=tc-neo4j-gen-measure --format '{{.Names}}' | grep -q . && { echo "a tc-neo4j-gen-measure container exists; remove it first"; exit 1; }

echo "---------------- 0 IMAGES"
MAIN_TAG=$($GI tag --build-commit "$MAIN_COMMIT")
GEN_TAG=$($GI tag --build-commit "$GEN_COMMIT")
if [ "$REBUILD" = 1 ]; then
    $GI remove --build-commit "$MAIN_COMMIT"
    [ "$GEN_COMMIT" = "$MAIN_COMMIT" ] || $GI remove --build-commit "$GEN_COMMIT"
fi
echo "---------------- 0a BUILD $MAIN_TAG (main) $(date)"
$GI ensure --build-commit "$MAIN_COMMIT" --torchcell-src "$TORCHCELL_SRC" 2>&1 | tee "$MEASURE_ROOT/build_main.log"
grep -oE '^IMAGE_BUILD_SECONDS=[0-9]+' "$MEASURE_ROOT/build_main.log" | sed 's/^/MAIN_/'
$GI check --build-commit "$MAIN_COMMIT"
docker image inspect --format 'main image size bytes: {{.Size}}' "$MAIN_TAG"
if [ "$GEN_COMMIT" != "$MAIN_COMMIT" ]; then
    echo "---------------- 0b BUILD $GEN_TAG (generator commit) $(date)"
    $GI ensure --build-commit "$GEN_COMMIT" --torchcell-src "$TORCHCELL_SRC" 2>&1 | tee "$MEASURE_ROOT/build_gen.log"
    grep -oE '^IMAGE_BUILD_SECONDS=[0-9]+' "$MEASURE_ROOT/build_gen.log" | sed 's/^/GEN_/'
    $GI check --build-commit "$GEN_COMMIT"
fi

# One generation of the adapter; $1 = run label, $2 = install|image.
generate() {
    # one assignment per line: bash expands every word of a `local` before it assigns any
    local label="$1"
    local mode="$2"
    local out_name="run_${label}_${mode}"
    local out_cont="/var/lib/neo4j/biocypher-out/$out_name/adapters/$ADAPTER_INDEX"
    local name="tc-neo4j-gen-measure-$label"
    local run_image install_cmd verify_cmd
    mkdir -p "$MEASURE_ROOT/$out_name/adapters"
    if [ "$mode" = install ]; then
        run_image="$IMAGE"
        install_cmd="python -m pip install --no-cache-dir --force-reinstall --no-deps '$PIP_REF' >/dev/null
        python -m pip install --no-cache-dir biocypher==0.15.2 >/dev/null
        python -m pip install --no-cache-dir git+https://github.com/oxpig/CaLM@main >/dev/null
        echo INSTALL_DONE"
        verify_cmd=":"
    else
        run_image="$GEN_TAG"
        install_cmd=":"
        verify_cmd="python - '$GEN_COMMIT' <<'PY'
import importlib.metadata as m, json, sys
got = json.loads(m.distribution('torchcell').read_text('direct_url.json'))['vcs_info']['commit_id']
sys.exit(0 if got == sys.argv[1] else f'image carries torchcell {got}, not {sys.argv[1]}')
PY
        echo COMMIT_VERIFIED"
    fi
    echo "---------------- RUN $label ($mode, $run_image) $(date)"
    {
    mark "T0_RUN_START run=$label mode=$mode image=$run_image"
    docker run --rm --entrypoint bash --name "$name" \
        --cpus="$CPUS" -e SLURM_CPUS_PER_TASK="$CPUS" --memory="${MEM_GB}g" \
        -e NEO4J_ACCEPT_LICENSE_AGREEMENT=yes -e PYTHONUNBUFFERED=1 \
        -e BUILD_COMMIT="$GEN_COMMIT" -e SLURM_JOB_ID="$JOB" -e SLURM_ARRAY_TASK_ID="$ADAPTER_INDEX" \
        -e TCDB_TAGS="gen-image-startup,${mode}" -e TCDB_JOB_TYPE="gen_image_startup" \
        -v "$MEASURE_ROOT":/var/lib/neo4j/biocypher-out \
        -v "$BUILD_ROOT/database/data":/var/lib/neo4j/data \
        -v "$DEV_DATA_ROOT/data/torchcell":/var/lib/neo4j/data/torchcell:ro \
        -v "$DEV_DATA_ROOT/data/ecoli":/var/lib/neo4j/data/ecoli:ro \
        -v "$DEV_DATA_ROOT/data/pputida":/var/lib/neo4j/data/pputida:ro \
        -v "$DEV_DATA_ROOT/torchcell-genomes":/var/lib/neo4j/torchcell-genomes:ro \
        -v "$BUILD_ROOT/database/.env":/.env \
        -v "$BUILD_ROOT/database/biocypher":/var/lib/neo4j/biocypher \
        "$run_image" -c "set -euo pipefail
        echo CONTAINER_SHELL_UP
        cd /var/lib/neo4j
        $install_cmd
        $verify_cmd
        set -a; source /.env; set +a
        wandb login \$WANDB_API_KEY >/dev/null
        mkdir -p $out_cont
        python -m torchcell.knowledge_graphs.generate_adapter --config-name $KG_CONFIG \
            --adapter-index $ADAPTER_INDEX --out-dir $out_cont --include-private --organisms $KG_ORGANISMS
        echo GENERATION_DIR=$out_cont" 2>&1
    mark "T1_RUN_END run=$label mode=$mode exit=$?"
    } | stamp | tee "$MEASURE_ROOT/$out_name.log"
    [ -f "$MEASURE_ROOT/$out_name/adapters/$ADAPTER_INDEX/DONE" ] || { echo "run $label ($mode) left no DONE marker"; exit 1; }
}

generate 0 install
for r in $(seq 1 "$RUNS"); do generate "$r" image; done

echo "---------------- TEARDOWN $(date)"
[ "$GEN_COMMIT" = "$MAIN_COMMIT" ] || $GI remove --build-commit "$GEN_COMMIT"
if [ "$KEEP_MAIN_IMAGE" = 1 ]; then echo "$MAIN_TAG kept for inspection (remove: $GI remove --build-commit $MAIN_COMMIT)"; else $GI remove --build-commit "$MAIN_COMMIT"; fi
docker images "tc-neo4j-gen" --format '{{.Repository}}:{{.Tag}} {{.Size}}'

echo "---------------- ANALYSIS $(date)"
python experiments/database/scripts/gen_image_startup_measurement.py \
    --out "$BUILD_ROOT/database/slurm/output/${JOB}_gen-image-measure.out" --job "$JOB" --baseline-seconds 278
echo "measurement finished at $(date)"
