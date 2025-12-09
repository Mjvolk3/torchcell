#!/bin/bash
#SBATCH --job-name=build_database
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=500G
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
    docker rm -f tc-neo4j || true

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
# HACK added sgd but this is because some of datasets need sgd.
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
    -v $(pwd)/database/data/sgd:/var/lib/neo4j/data/sgd \
    -v $(pwd)/database/data:/var/lib/neo4j/data \
    -v $(pwd)/database/.env:/.env \
    -v $(pwd)/database/conf:/var/lib/neo4j/conf \
    -v $(pwd)/database/logs:/logs \
    -v $(pwd)/database/plugins:/plugins \
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
docker exec tc-neo4j python -m pip install --no-cache-dir hypernetx fastjsonschema

echo "Setting up BioCypher configuration..."
docker exec tc-neo4j mkdir -p /var/lib/neo4j/biocypher/config
docker exec tc-neo4j wget -O /var/lib/neo4j/biocypher/config/torchcell_schema_config.yaml \
  https://raw.githubusercontent.com/Mjvolk3/torchcell/main/biocypher/config/torchcell_schema_config.yaml
docker exec tc-neo4j wget -O /var/lib/neo4j/biocypher/config/linux-amd_biocypher_config.yaml \
  https://raw.githubusercontent.com/Mjvolk3/torchcell/main/biocypher/config/linux-amd_biocypher_config.yaml

# Check what paths are in .env
echo "Checking .env configuration..."
docker exec tc-neo4j bash -c "grep -E 'BIOCYPHER_CONFIG_PATH|SCHEMA_CONFIG_PATH' /.env || echo 'Paths not found in .env'"

echo "Logging in to wandb..."
docker exec tc-neo4j wandb login $WANDB_API_KEY

echo "----------------NOW_BUILDING_GRAPHS---------------------"

#torchcell.knowledge_graphs.gene_interactions_scerevisae_kg
#torchcell.torchcell.knowledge_graphs.create_scerevisiae_kg
#torchcell.knowledge_graphs.dmi_costanzo_2016_kg
echo "Creating knowledge graph..."
if ! docker exec tc-neo4j bash -c "export BIOCYPHER_CONFIG_PATH=/var/lib/neo4j/biocypher/config/linux-amd_biocypher_config.yaml && \
  export SCHEMA_CONFIG_PATH=/var/lib/neo4j/biocypher/config/torchcell_schema_config.yaml && \
  export DATA_ROOT=/var/lib/neo4j && \
  export BIOCYPHER_OUT_PATH=biocypher-out && \
  python -m torchcell.knowledge_graphs.create_kg --config-name=calmorph_ohya2005_kg"; then
    echo "Error: Failed to create knowledge graph"
    docker logs tc-neo4j
    exit 1
fi
#scerevisiae_global_kg
#calmorph_ohya2005_kg

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
docker exec tc-neo4j bash -c "source /.env && cypher-shell -u \"\$NEO4J_USER\" -p \"\$NEO4J_PASSWORD\" -d torchcell 'MATCH (n) RETURN count(n);'"

# Verify APOC installation
echo "Verifying APOC installation..."
docker exec tc-neo4j bash -c "source /.env && cypher-shell -u \"\$NEO4J_USER\" -p \"\$NEO4J_PASSWORD\" -d torchcell 'RETURN apoc.version() AS apoc_version;'"

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