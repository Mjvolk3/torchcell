#!/bin/bash
#SBATCH --job-name=build_database
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/projects/torchcell/database/slurm/output/%j_%x.out
#SBATCH --error=/scratch/projects/torchcell/database/slurm/output/%j_%x.out
#SBATCH --no-requeue

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
docker rm -f tc-neo4j || true

# Ensure Docker uses the SLURM cgroup
echo "Starting Docker container within SLURM cgroup..."

docker run \
    --cpus=$SLURM_CPUS_PER_TASK \
    --memory=${mem_mb}m \
    --env=NEO4J_ACCEPT_LICENSE_AGREEMENT=yes \
    -d --name tc-neo4j \
    -p 7474:7474 -p 7687:7687 \
    -v $(pwd)/database/biocypher-out:/var/lib/neo4j/biocypher-out:Z \
    -v $(pwd)/data/torchcell:/var/lib/neo4j/data/torchcell:Z \
    -v $(pwd)/database/data:/var/lib/neo4j/data:Z \
    -v $(pwd)/database/.env:/.env:Z \
    -v $(pwd)/database/biocypher:/var/lib/neo4j/biocypher:Z \
    -v $(pwd)/database/conf:/var/lib/neo4j/conf:Z \
    -v $(pwd)/database/logs:/logs:Z \
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
docker exec tc-neo4j python -m pip install --no-cache-dir git+https://github.com/Mjvolk3/torchcell.git@main
docker exec tc-neo4j python -m pip install --no-cache-dir git+https://github.com/Mjvolk3/biocypher@main
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

echo "Executing generated bash script..."
docker exec tc-neo4j bash -c 'if [ -f biocypher_file_name.txt ]; then 
    bash_script_path_cleaned=$(cat biocypher_file_name.txt)
    chmod +x "${bash_script_path_cleaned}"
    "${bash_script_path_cleaned}"
else 
    echo "biocypher_file_name.txt not found"
fi'

echo "Protecting directory..."
docker exec tc-neo4j bash -c 'if [ -f biocypher_file_name.txt ]; then 
    dir_path=$(dirname $(cat biocypher_file_name.txt))
    chmod a-w "${dir_path}"
else 
    echo "biocypher_file_name.txt not found"
fi'

echo "Starting Neo4j service..."
docker exec tc-neo4j neo4j start

echo "Copying neo4j.conf to host..."
docker cp tc-neo4j:/var/lib/neo4j/conf/neo4j.conf /scratch/projects/torchcell/database/conf/neo4j.conf

echo "Copying biocypher config to host..."
docker cp tc-neo4j:/var/lib/neo4j/biocypher/config /scratch/projects/torchcell/database/biocypher/

echo "Ensuring files are readable..."
chmod 644 /scratch/projects/torchcell/database/conf/neo4j.conf
chmod -R 644 /scratch/projects/torchcell/database/biocypher/config

echo "File locations:"
echo "neo4j.conf: /scratch/projects/torchcell/database/conf/neo4j.conf"
echo "biocypher config: /scratch/projects/torchcell/database/biocypher/config/"

echo "Build and run process completed."