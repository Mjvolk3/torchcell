#!/bin/bash
#SBATCH --job-name=build_database
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=995G
#SBATCH --time=365-00:00:00
#SBATCH --output=/scratch/projects/torchcell/database/slurm/output/%j_%x.out
#SBATCH --error=/scratch/projects/torchcell/database/slurm/output/%j_%x.out
#SBATCH --no-requeue

###
# Allow for working with docker in slurm env
# Cancel docker with scancel
cleanup() {
    echo "Received termination signal. Cleaning up..."
    
    # Stop and remove the Docker container
    echo "Stopping and removing Docker container..."
    docker stop tc-neo4j || true

    echo "Cleanup completed. Exiting."
    exit 0
}
trap cleanup SIGTERM SIGINT SIGHUP EXIT
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
    -v $(pwd)/database/plugins:/plugins \
    -v $(pwd)/database/import:/import \
    -v $(pwd)/database/metrics:/metrics \
    -e NEO4J_AUTH=neo4j/torchcell \
    -e NEO4J_dbms_read__only=false \
    -e NEO4J_apoc_export_file_enabled=true \
    -e NEO4J_apoc_import_file_enabled=true \
    -e NEO4J_apoc_import_file_use__neo4j__config=true \
    -e NEO4J_dbms_security_procedures_unrestricted=apoc.\* \
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
docker exec tc-neo4j python -m pip install --no-cache-dir --force-reinstall --no-deps git+https://github.com/Mjvolk3/torchcell.git@main
sleep 5
docker exec tc-neo4j python -m pip install --no-cache-dir git+https://github.com/Mjvolk3/biocypher@main
sleep 5
docker exec tc-neo4j python -m pip install --no-cache-dir git+https://github.com/oxpig/CaLM@main


echo "Logging in to wandb..."
docker exec tc-neo4j wandb login $WANDB_API_KEY

docker exec tc-neo4j neo4j stop
docker exec tc-neo4j neo4j start


### Slurm-Docker interrupt
while true; do
    if ! docker ps -q --filter name=tc-neo4j; then
        echo "Docker container has stopped. Exiting."
        cleanup
    fi
    sleep 10
done
###