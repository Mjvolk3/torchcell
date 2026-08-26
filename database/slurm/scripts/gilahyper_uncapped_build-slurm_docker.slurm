#!/bin/bash
#SBATCH --job-name=build_database
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH --time=10-00:00:00
#SBATCH --output=/scratch/projects/torchcell/database/slurm/output/%j_%x.out
#SBATCH --error=/scratch/projects/torchcell/database/slurm/output/%j_%x.out
#SBATCH --no-requeue

# The UNCAPPED build: every dataset in full, both Costanzo 2016 double-mutant sets at
# their full 20,705,612 records each (config kg_uncapped). Same pipeline as
# gilahyper_first_build-slurm_docker.slurm; only the resource envelope and the default
# KG_CONFIG differ, because this build is ~440x the record count of the capped one.
#
#   sbatch database/slurm/scripts/gilahyper_uncapped_build-slurm_docker.slurm
#
# RESOURCES, and why they are what they are. GilaHyper is 128 CPU / 513 G.
#   time  10 days. Projected CSV generation is 66.9 h at the job-966 rates, but that
#         projection charges dmf Costanzo a rate measured while its loader root still
#         pointed at the smf LMDB; at the dmi rate the projection is ~108 h. Bulk
#         import follows generation. 10 days covers the spread with margin.
#   cpus  32, matching job 970 (the last capped kg_full build) exactly. More workers
#         would generate faster -- CellAdapter.get_data_by_type maps chunks over a
#         ProcessPoolExecutor -- but it submits EVERY chunk up front and yields results
#         in submission order, so finished-but-unyielded chunk results accumulate in
#         the parent. At 100k records each Costanzo adapter was a single chunk and
#         never buffered; at 20.7M it is ~52 chunks (~104 at the 0.5 memory-reduction
#         methods), and faster workers mean a deeper buffer. Raising this trades
#         wall time against a memory profile nothing has yet measured at this scale.
#         To take that trade: sbatch --cpus-per-task=64 <this script>.
#   mem   480 G, up from job 970's 256 G. Pure headroom for the buffering above; it
#         costs nothing to reserve on an otherwise idle node.
#
# DISK. Job 970 wrote 50.3 GB of CSV with the doubles capped at 100k, of which their
# Experiment nodes were ~10.2 KB per record. Scaling that to 41.2M more records puts
# generation near 600 GB of CSV, and the neo4j store import lands on the same
# filesystem. Check `df -h /scratch` before launching and clear stale
# database/biocypher-out/<timestamp> trees if it is close.

echo "Running on $(hostname)"

###
source ~/.zshrc # Source a minimal environment, adjust to your shell if needed
conda activate ~/miniconda3/envs/torchcell
# directory_setup builds DATA_ROOT/database/... and copies conf/gh_neo4j.conf +
# database.env in. The BUILD TREE is the mounted drive /scratch/projects/torchcell
# (what the docker run below bind-mounts), NOT the dev tree in the host .env
# (torchcell-scratch). load_dotenv(override=False) keeps this explicit DATA_ROOT.
DATA_ROOT=/scratch/projects/torchcell python -m torchcell.database.directory_setup
###

echo "Checking Docker availability..."
which docker
docker --version

echo "Slurm environment:"
echo "SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK"
echo "SLURM_MEM_PER_NODE: $SLURM_MEM_PER_NODE"

# Convert SLURM_MEM_PER_NODE to megabytes
mem_mb=$((SLURM_MEM_PER_NODE))
echo "Memory in MB: $mem_mb"

echo "Free space on the build filesystem:"
df -h /scratch

# Move to the project directory
cd /scratch/projects/torchcell

# Remove existing container if it exists
docker stop tc-neo4j || true
docker stop tc-neo4j-readonly || true
docker rm -f tc-neo4j || true
docker rm -f tc-neo4j-readonly || true

# Ensure Docker uses the SLURM cgroup
echo "Starting Docker container within SLURM cgroup..."

# We don't forward ports for the writable container, as it will be stopped after the database is created
docker run \
    --cpus=$SLURM_CPUS_PER_TASK \
    -e SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK \
    --memory=${mem_mb}m \
    --env=NEO4J_ACCEPT_LICENSE_AGREEMENT=yes \
    -d --name tc-neo4j \
    -p 7687:7687 -p 7474:7474 \
    --restart=unless-stopped \
    -v $(pwd)/database/biocypher-out:/var/lib/neo4j/biocypher-out \
    -v $(pwd)/database/data/torchcell:/var/lib/neo4j/data/torchcell \
    -v $(pwd)/database/data:/var/lib/neo4j/data \
    -v $(pwd)/database/.env:/.env \
    -v $(pwd)/database/biocypher:/var/lib/neo4j/biocypher \
    -v $(pwd)/database/conf:/var/lib/neo4j/conf \
    -v $(pwd)/database/logs:/logs \
    -v $(pwd)/database/metrics:/metrics \
    -e NEO4J_AUTH=neo4j/torchcell \
    -e NEO4J_server_databases_default__to__read__only=false \
    michaelvolk/tc-neo4j:5.26.28

# Check if the container is running
if [ $(docker ps -q -f name=tc-neo4j) ]; then
    echo "Container is running"
else
    echo "Container failed to start"
    docker logs tc-neo4j
    exit 1
fi

echo "Updating software in the container..."

# The container installs torchcell from github main, so kg_uncapped.yaml (shipped via
# the knowledge_graphs/conf/*.yaml package-data glob) must be ON MAIN before launch.
sleep 5
docker exec tc-neo4j python -m pip install --no-cache-dir --force-reinstall --no-deps git+https://github.com/Mjvolk3/torchcell.git@main
sleep 5
docker exec tc-neo4j python -m pip install --no-cache-dir biocypher==0.15.2
sleep 5
docker exec tc-neo4j python -m pip install --no-cache-dir git+https://github.com/oxpig/CaLM@main

echo "Verifying the requested config shipped with the installed torchcell..."
KG_CONFIG="${KG_CONFIG:-kg_uncapped}"
if ! docker exec tc-neo4j python -c "import importlib.resources as r, sys; p = r.files('torchcell').joinpath('knowledge_graphs/conf/${KG_CONFIG}.yaml'); sys.exit(0 if p.is_file() else 1)"; then
    echo "Error: ${KG_CONFIG}.yaml is not in the installed torchcell -- has it landed on main?"
    exit 1
fi

echo "Logging in to wandb..."
docker exec tc-neo4j bash -c 'source /.env && wandb login $WANDB_API_KEY'

echo "----------------NOW_BUILDING_GRAPHS---------------------"

echo "Creating knowledge graph..."

# Parameterized the same way as gilahyper_first_build-slurm_docker.slurm; only the
# KG_CONFIG default differs (kg_uncapped instead of kg_small).
KG_MODULE="${KG_MODULE:-torchcell.knowledge_graphs.create_scerevisiae_kg_small}"
echo "Building KG: module=$KG_MODULE config=$KG_CONFIG"
# source /.env INSIDE the container so DATA_ROOT et al. are in the environment before
# python imports torchcell -- several torchcell modules (e.g. torchcell.graph.sgd) read
# DATA_ROOT at import time, before the build script's own load_dotenv runs.
if ! docker exec tc-neo4j bash -c "set -a && source /.env && set +a && python -m $KG_MODULE --config-name $KG_CONFIG"; then
    echo "Error: Failed to create knowledge graph"
    docker logs tc-neo4j
    exit 1
fi

echo "Knowledge graph creation completed."
echo "Free space after CSV generation:"
df -h /scratch

# neo4j 5 refuses to bulk-import into a database that exists and is online
# ("The database is in use. Stop database 'torchcell' and try again."), which neo4j 4.4
# did not enforce. Dropping it also clears the "Database name or alias already exists"
# failure that CREATE DATABASE hits on every rebuild.
echo "Dropping any existing torchcell database before import..."
docker exec tc-neo4j bash -c 'source /.env && cypher-shell -u "$NEO4J_USER" -p "$NEO4J_PASSWORD" "DROP DATABASE torchcell IF EXISTS;"'

echo "Executing generated bash script..."
docker exec tc-neo4j bash -c 'if [ -f biocypher_file_name.txt ]; then
    bash_script_path_cleaned=$(cat biocypher_file_name.txt)
    chmod +x "${bash_script_path_cleaned}"
    "${bash_script_path_cleaned}"
else
    echo "biocypher_file_name.txt not found"
fi'

echo "Starting Neo4j service..."
docker exec tc-neo4j neo4j start

docker exec tc-neo4j sleep 180
docker exec tc-neo4j bash -c 'source /.env && cypher-shell -u "$NEO4J_USER" -p "$NEO4J_PASSWORD" "CREATE DATABASE torchcell;"'

echo "Running query to generate index stats..."
docker exec tc-neo4j neo4j stop
docker exec tc-neo4j neo4j start

# sleep for db to come up before query
sleep 60

# We must run a query to generate the index-stats file the readonly container needs.
# neo4j 5.26 Enterprise defaults to BLOCK store format, so this is
# database/data/databases/torchcell/block.indexstats.db (4.4 wrote neostore.indexstats.db).
docker exec tc-neo4j bash -c 'source /.env && cypher-shell -u $NEO4J_USER -p $NEO4J_PASSWORD -d torchcell "MATCH (n) RETURN count(n);"'

echo "Stopping the writable container..."
docker stop tc-neo4j

echo "Creating read-only container..."
docker run \
    --cpus=$SLURM_CPUS_PER_TASK \
    --memory=${mem_mb}m \
    --env=NEO4J_ACCEPT_LICENSE_AGREEMENT=yes \
    -d --name tc-neo4j-readonly \
    -p 7687:7687 -p 7474:7474 \
    --health-cmd="curl -f http://gilahyper.zapto.org:7474 || exit 1" \
    --health-interval=30s \
    --health-timeout=10s \
    --health-retries=3 \
    --restart=unless-stopped \
    -v $(pwd)/database/biocypher-out:/var/lib/neo4j/biocypher-out \
    -v $(pwd)/data/torchcell:/var/lib/neo4j/data/torchcell \
    -v $(pwd)/database/data:/var/lib/neo4j/data \
    -v $(pwd)/database/.env:/.env \
    -v $(pwd)/database/biocypher:/var/lib/neo4j/biocypher \
    -v $(pwd)/database/conf:/var/lib/neo4j/conf \
    -v $(pwd)/database/logs:/logs \
    -v $(pwd)/database/metrics:/metrics \
    -e NEO4J_AUTH=neo4j/torchcell \
    -e NEO4J_server_databases_default__to__read__only=true \
    michaelvolk/tc-neo4j:5.26.28

echo "Waiting for read-only Neo4j to start..."
