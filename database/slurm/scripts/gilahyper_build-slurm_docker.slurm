#!/bin/bash
#SBATCH --job-name=build_database
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/projects/torchcell/database/slurm/output/%j_%x.out
#SBATCH --error=/scratch/projects/torchcell/database/slurm/output/%j_%x.out
#SBATCH --no-requeue

###
# Allow for working with docker in slurm env
# Cancel docker with scancel
cleanup() {
    echo "Stopping Docker container..."
    docker stop tc-neo4j
    docker rm tc-neo4j
    exit 0
}

trap cleanup SIGTERM SIGINT
###

echo "Running on $(hostname)"

echo "Checking Docker availability..."
which docker
docker --version

echo "Slurm environment:"
echo "SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK"
echo "SLURM_MEM_PER_NODE: $SLURM_MEM_PER_NODE"

# Convert SLURM_MEM_PER_NODE to megabytes
mem_mb=$((SLURM_MEM_PER_NODE))
echo "Memory in MB: $mem_mb"

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
    --cgroup-parent=/slurm/uid_${SLURM_JOB_UID}/job_${SLURM_JOB_ID}/step_${SLURM_STEP_ID}/task_${SLURM_TASK_PID} \
    --cpus=$SLURM_CPUS_PER_TASK \
    --memory=${mem_mb}m \
    --env=NEO4J_ACCEPT_LICENSE_AGREEMENT=yes \
    -d --name tc-neo4j \
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
    -e NEO4J_AUTH=neo4j/torchcell \
    michaelvolk/tc-neo4j:latest

# Check if the container is running
if [ $(docker ps -q -f name=tc-neo4j) ]; then
    echo "Container is running"
else
    echo "Container failed to start"
    docker logs tc-neo4j
    exit 1
fi

echo "Updating software in the container..."

sleep 5
docker exec tc-neo4j python -m pip install --no-cache-dir git+https://github.com/Mjvolk3/torchcell.git@main
sleep 5
docker exec tc-neo4j python -m pip install --no-cache-dir git+https://github.com/Mjvolk3/biocypher@main
sleep 5
docker exec tc-neo4j python -m pip install --no-cache-dir git+https://github.com/oxpig/CaLM@main


echo "Logging in to wandb..."
docker exec tc-neo4j wandb login $WANDB_API_KEY

echo "----------------NOW_BUILDING_GRAPHS---------------------"

echo "Creating knowledge graph..."
if ! docker exec tc-neo4j python -m torchcell.knowledge_graphs.minimal_kg; then
    echo "Error: Failed to create knowledge graph"
    docker logs tc-neo4j
    exit 1
fi

echo "Knowledge graph creation completed."


# If database exists we need to stop it for bulk import.
docker exec tc-neo4j bash -c 'source /.env && cypher-shell -u "$NEO4J_USER" -p "$NEO4J_PASSWORD" "STOP DATABASE torchcell;"'

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
# create the database - will fail if exists...
docker exec tc-neo4j bash -c 'source /.env && cypher-shell -u "$NEO4J_USER" -p "$NEO4J_PASSWORD" "CREATE DATABASE torchcell;"'

# start database - needed if exists but stopped
docker exec tc-neo4j bash -c 'source /.env && cypher-shell -u "$NEO4J_USER" -p "$NEO4J_PASSWORD" "START DATABASE torchcell;"'

echo "Running query to generate index stats..."
docker exec tc-neo4j bash -c 'source /.env && cypher-shell -u $NEO4J_USER -p $NEO4J_PASSWORD -d torchcell "MATCH (n) RETURN count(n);"'

docker exec tc-neo4j neo4j stop
docker exec tc-neo4j neo4j start

echo "All setup completed. Keeping SLURM job alive to maintain resource control..."

while true; do
    if ! docker ps -q --filter name=tc-neo4j; then
        echo "Docker container has stopped. Exiting."
        exit 0
    fi
    sleep 10
done