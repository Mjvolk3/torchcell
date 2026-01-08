#!/bin/bash
#SBATCH --job-name=run_neo4j
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/projects/torchcell-scratch/database/slurm/output/%j_%x.out
#SBATCH --error=/scratch/projects/torchcell-scratch/database/slurm/output/%j_%x.out
#SBATCH --no-requeue

# =============================================================================
# Run Neo4j Container (No Build)
# =============================================================================
# This script starts the Neo4j container with existing data for:
#   - Exporting the database
#   - Running queries
#   - General database access
#
# Usage:
#   sbatch gilahyper_run_container-slurm_docker.slurm
#
# Then in another terminal:
#   bash database/scripts/export_database.sh
#
# To stop: scancel <job_id>
# =============================================================================

###
# Cleanup on exit - stops and removes container
cleanup() {
    echo ""
    echo "Received termination signal. Cleaning up..."
    echo "Stopping and removing Docker container..."
    docker stop tc-neo4j || true
    docker rm -f tc-neo4j || true
    echo "Cleanup completed. Exiting."
    exit 0
}
trap cleanup SIGTERM SIGINT SIGHUP EXIT
###

echo "========================================="
echo "Starting Neo4j Container (Run Mode)"
echo "========================================="
echo "Running on $(hostname)"
echo "Started at: $(date)"
echo ""

echo "Checking Docker availability..."
which docker
docker --version

# Move to the project directory
cd /scratch/projects/torchcell-scratch

# Remove existing container if it exists
echo "Removing any existing containers..."
docker stop tc-neo4j 2>/dev/null || true
docker rm -f tc-neo4j 2>/dev/null || true

echo ""
echo "Starting Docker container..."

docker run \
    --cpus=${SLURM_CPUS_PER_TASK:-8} \
    --memory=${SLURM_MEM_PER_NODE:-65536}m \
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
    echo ""
    echo "========================================="
    echo "Container is running!"
    echo "========================================="
    echo ""
    echo "Neo4j is starting up (takes ~30-60 seconds)..."
    echo ""
    echo "You can now run in another terminal:"
    echo "  bash database/scripts/export_database.sh"
    echo ""
    echo "Or connect to Neo4j Browser:"
    echo "  http://$(hostname):7474"
    echo ""
    echo "To stop this job: scancel $SLURM_JOB_ID"
    echo "========================================="
else
    echo "Container failed to start"
    docker logs tc-neo4j
    exit 1
fi

# Wait for Neo4j to be ready
echo ""
echo "Waiting for Neo4j to be ready..."
sleep 30

# Verify Neo4j is responding
if docker exec tc-neo4j bash -c 'source /.env && cypher-shell -u "$NEO4J_USER" -p "$NEO4J_PASSWORD" "RETURN 1;"' 2>/dev/null; then
    echo "Neo4j is ready and responding!"
else
    echo "Warning: Neo4j may still be starting up. Wait a bit longer before running exports."
fi

echo ""
echo "Container running. Monitoring..."
echo ""

### Keep job alive while container runs
while true; do
    if ! docker ps -q --filter name=tc-neo4j | grep -q .; then
        echo "Docker container has stopped. Exiting."
        cleanup
    fi
    sleep 30
done
###
