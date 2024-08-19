#!/bin/bash
#SBATCH --job-name=neo4j_readonly
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=365-0:00:00
#SBATCH --output=/scratch/projects/torchcell/database/slurm/output/%j_%x.out
#SBATCH --error=/scratch/projects/torchcell/database/slurm/output/%j_%x.out
#SBATCH --no-requeue

###
# Allow for working with docker in slurm env
# Cancel docker with scancel
cleanup() {
    echo "Stopping Docker container..."
    docker stop tc-neo4j-readonly
    docker rm tc-neo4j-readonly
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

# Remove existing read-only container if it exists
docker stop tc-neo4j-readonly || true
docker rm -f tc-neo4j-readonly || true

echo "Starting read-only Docker container within SLURM cgroup..."

docker run \
    --cgroup-parent=/slurm/uid_${SLURM_JOB_UID}/job_${SLURM_JOB_ID}/step_${SLURM_STEP_ID}/task_${SLURM_TASK_PID} \
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
    -e NEO4J_dbms_read__only=true \
    -e NEO4J_apoc_export_file_enabled=true \
    -e NEO4J_apoc_import_file_enabled=true \
    -e NEO4J_apoc_import_file_use__neo4j__config=true \
    -e NEO4J_dbms_security_procedures_unrestricted=apoc.\* \
    michaelvolk/tc-neo4j:latest

# Check if the container is running
if [ $(docker ps -q -f name=tc-neo4j-readonly) ]; then
    echo "Read-only container is running"
else
    echo "Read-only container failed to start"
    docker logs tc-neo4j-readonly
    exit 1
fi

echo "Waiting for read-only Neo4j to start..."
sleep 60

echo "Running test query..."
docker exec tc-neo4j-readonly bash -c 'source /.env && cypher-shell -u $NEO4J_USER -p $NEO4J_PASSWORD -d torchcell "MATCH (n) RETURN count(n);"'

echo "Read-only Neo4j container is now running and ready for use."

echo "All setup completed. Keeping SLURM job alive to maintain resource control..."

while true; do
    if ! docker ps -q --filter name=tc-neo4j-readonly; then
        echo "Docker container has stopped. Exiting."
        exit 0
    fi
    sleep 10
done