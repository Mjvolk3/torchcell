#!/bin/bash
#SBATCH --job-name=build_database
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=980G
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/projects/torchcell/database/slurm/output/%j_%x.out
#SBATCH --error=/scratch/projects/torchcell/database/slurm/output/%j_%x.out
#SBATCH --no-requeue

##SBATCH --mem=64G
##SBATCH --time=12:00:00

echo "Running on $(hostname)"

###
source ~/.zshrc # Source a minimal environment, adjust to your shell if needed
conda activate ~/miniconda3/envs/torchcell
python -m torchcell.database.directory_setup
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

# Move to the project directory
cd /scratch/projects/torchcell

# Remove existing container if it existso
docker stop tc-neo4j || true
docker stop tc-neo4j-readonly || true
docker rm -f tc-neo4j || true
docker rm -f tc-neo4j-readonly || true

# Ensure Docker uses the SLURM cgroup
echo "Starting Docker container within SLURM cgroup..."

# We don't forward ports for the writable container, as it will be stopped after the database is created
#    -e OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK \
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
    -e NEO4J_AUTH=neo4j/torchcell \
    -e NEO4J_dbms_read__only=false \
    michaelvolk/tc-neo4j:latest

#--health-cmd="curl -f http://gilahyper.zapto.org:7474 || exit 1" \
#--health-interval=30s \
#--health-timeout=10s \
#--health-retries=3 \

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
docker exec tc-neo4j bash -c 'source /.env && wandb login $WANDB_API_KEY'

echo "----------------NOW_BUILDING_GRAPHS---------------------"

echo "Creating knowledge graph..."

#torchcell.knowledge_graphs.minimal_kg
#torchcell.knowledge_graphs.create_scerevisiae_kg
#torchcell.knowledge_graphs.gene_interactions_scerevisae_kg_small
#torchcell.knowledge_graphs.dmi_kuzmin_2018_kg
#torchcell.knowledge_graphs.dmi_costanzo_2016_kg
if ! docker exec tc-neo4j python -m torchcell.knowledge_graphs.tmi_kuzmin_2018_kg; then
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

echo "Starting Neo4j service..."
docker exec tc-neo4j neo4j start

docker exec tc-neo4j sleep 180
docker exec tc-neo4j bash -c 'source /.env && cypher-shell -u "$NEO4J_USER" -p "$NEO4J_PASSWORD" "CREATE DATABASE torchcell;"'

echo "Running query to generate index stats..."
docker exec tc-neo4j neo4j stop
docker exec tc-neo4j neo4j start

# sleep for db to come up before query
sleep 60

# We must run a query to get the database/data/databases/torchcell/neostore.indexstats.db file which is needed for the readonly container.
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
    -e NEO4J_AUTH=neo4j/torchcell \
    -e NEO4J_dbms_read__only=true \
    michaelvolk/tc-neo4j:latest


echo "Waiting for read-only Neo4j to start..."

#sudo setfacl -R -m u:$(whoami):rx /scratch/projects/torchcell/database/biocypher /scratch/projects/torchcell/database/biocypher-out /scratch/projects/torchcell/database/conf /scratch/projects/torchcell/database/data /scratch/projects/torchcell/database/logs