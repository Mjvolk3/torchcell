#!/bin/bash
# Neo4j Database Import Script for OpenStack
# Imports a database dump from gilahyper build

set -e

PROJECT_DIR="/home/rocky/projects/torchcell"
cd "$PROJECT_DIR" || exit 1

# Configuration
CONTAINER_NAME="tc-neo4j"
DATABASE_NAME="torchcell"
IMPORT_DIR="/mnt/delta_bbub/mjvolk3/torchcell/database/import"
LOCAL_DATA_DIR="$HOME/neo4j-data"
DELTA_DATA_DIR="/mnt/delta_bbub/mjvolk3/torchcell"

# Initial password for container operations
INITIAL_PASSWORD="torchcell"

# Read final password from YAML config
CONFIG_FILE="$PROJECT_DIR/biocypher/config/PRODUCTION_linux-amd_biocypher_config.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found at $CONFIG_FILE"
    exit 1
fi
YAML_PASSWORD=$(grep "password:" "$CONFIG_FILE" | tail -1 | awk '{print $2}')

if [ -z "$YAML_PASSWORD" ]; then
    echo "Error: Could not extract password from $CONFIG_FILE"
    exit 1
fi

echo "========================================="
echo "Neo4j Database Import Script"
echo "========================================="

# Check for dump file argument or use latest
if [ $# -eq 0 ]; then
    DUMP_FILE="${IMPORT_DIR}/${DATABASE_NAME}_latest.dump"
    echo "No dump file specified, using latest: ${DUMP_FILE}"
else
    DUMP_FILE="$1"
    # If relative path, prepend import directory
    if [[ ! "$DUMP_FILE" = /* ]]; then
        DUMP_FILE="${IMPORT_DIR}/${DUMP_FILE}"
    fi
fi

# Verify dump file exists (check with sudo if needed)
if [ ! -f "${DUMP_FILE}" ] && ! sudo test -f "${DUMP_FILE}"; then
    echo "Error: Dump file not found: ${DUMP_FILE}"
    echo ""
    echo "Available dumps in ${IMPORT_DIR}:"
    sudo ls -lh "${IMPORT_DIR}/"*.dump 2>/dev/null || echo "No dump files found"
    exit 1
fi

# Check if we need sudo to access the file
if [ ! -r "${DUMP_FILE}" ]; then
    echo "Note: Using sudo to access dump file due to permissions"
    NEED_SUDO=true
else
    NEED_SUDO=false
fi

echo "Database: ${DATABASE_NAME}"
echo "Import from: ${DUMP_FILE}"
if [ "$NEED_SUDO" = true ]; then
    echo "File size: $(sudo ls -lh ${DUMP_FILE} | awk '{print $5}')"
else
    echo "File size: $(ls -lh ${DUMP_FILE} | awk '{print $5}')"
fi
echo ""

# Check if container is running, start if not
if docker ps -q -f name=${CONTAINER_NAME} > /dev/null 2>&1; then
    echo "✓ Container ${CONTAINER_NAME} is running"
else
    echo "Starting ${CONTAINER_NAME} container..."
    
    # Ensure mount is available
    if ! mountpoint -q /mnt/delta_bbub; then
        echo "Mounting Delta storage..."
        sudo mount -t nfs taiga-nfs.ncsa.illinois.edu:/taiga/nsf/delta/bbub /mnt/delta_bbub
    fi
    
    # Clean up old local data
    if [ -d "$LOCAL_DATA_DIR" ]; then
        echo "Cleaning up old local data..."
        sudo rm -rf $LOCAL_DATA_DIR
    fi
    mkdir -p $LOCAL_DATA_DIR/{neo4j-data,logs}
    chmod -R 777 $LOCAL_DATA_DIR
    
    # Start container
    docker run \
        --cpus=14 \
        --memory=56g \
        --tmpfs /tmp:size=10G \
        --env=NEO4J_ACCEPT_LICENSE_AGREEMENT=yes \
        -d --name ${CONTAINER_NAME} \
        -p 7474:7474 -p 7473:7473 -p 7687:7687 \
        -v "$DELTA_DATA_DIR/biocypher-out:/var/lib/neo4j/biocypher-out" \
        -v "$DELTA_DATA_DIR/torchcell:/var/lib/neo4j/data/torchcell" \
        -v "$DELTA_DATA_DIR/sgd:/var/lib/neo4j/data/sgd" \
        -v "$LOCAL_DATA_DIR/neo4j-data:/var/lib/neo4j/data" \
        -v "$(pwd)/database/.env:/.env:ro" \
        -v "$(pwd)/biocypher:/var/lib/neo4j/biocypher" \
        -v "$(pwd)/database/conf:/var/lib/neo4j/conf" \
        -v "$(pwd)/database/certificates:/var/lib/neo4j/certificates" \
        --tmpfs /logs:size=1G \
        -v "$(pwd)/database/plugins:/plugins" \
        -e NEO4J_AUTH=neo4j/${INITIAL_PASSWORD} \
        -e NEO4J_dbms_read__only=false \
        -e NEO4J_apoc_export_file_enabled=true \
        -e NEO4J_apoc_import_file_enabled=true \
        -e NEO4J_apoc_import_file_use__neo4j__config=true \
        -e NEO4J_dbms_security_procedures_unrestricted=apoc.\\* \
        michaelvolk/tc-neo4j:latest
    
    echo "Waiting for container to start..."
    sleep 30
    
    if ! docker ps | grep -q ${CONTAINER_NAME}; then
        echo "Container failed to start. Checking logs..."
        docker logs --tail 50 ${CONTAINER_NAME}
        exit 1
    fi
fi

echo ""
echo "Step 1: Creating dumps directory in container..."
docker exec ${CONTAINER_NAME} mkdir -p /var/lib/neo4j/data/dumps

echo "Step 2: Copying dump file to container..."
echo "  This may take a while for large files..."
if [ "$NEED_SUDO" = true ]; then
    # Create a temporary copy with proper permissions
    TEMP_DUMP="/tmp/import_$(basename ${DUMP_FILE})"
    echo "  Creating temporary copy with proper permissions..."
    sudo cp "${DUMP_FILE}" "${TEMP_DUMP}"
    sudo chmod 644 "${TEMP_DUMP}"
    docker cp "${TEMP_DUMP}" ${CONTAINER_NAME}:/var/lib/neo4j/data/dumps/import.dump
    sudo rm -f "${TEMP_DUMP}"
else
    docker cp "${DUMP_FILE}" ${CONTAINER_NAME}:/var/lib/neo4j/data/dumps/import.dump
fi

echo "Step 3: Stopping the database if it exists..."
docker exec ${CONTAINER_NAME} cypher-shell -u neo4j -p "${INITIAL_PASSWORD}" -d system "STOP DATABASE ${DATABASE_NAME};" 2>/dev/null || \
docker exec ${CONTAINER_NAME} cypher-shell -u neo4j -p "${YAML_PASSWORD}" -d system "STOP DATABASE ${DATABASE_NAME};" 2>/dev/null || true
sleep 5

echo "Step 4: Dropping the database if it exists..."
docker exec ${CONTAINER_NAME} cypher-shell -u neo4j -p "${INITIAL_PASSWORD}" -d system "DROP DATABASE ${DATABASE_NAME} IF EXISTS;" 2>/dev/null || \
docker exec ${CONTAINER_NAME} cypher-shell -u neo4j -p "${YAML_PASSWORD}" -d system "DROP DATABASE ${DATABASE_NAME} IF EXISTS;" 2>/dev/null || true
sleep 5

echo "Step 5: Loading database from dump..."
# Create a temp directory that allows execution for zstd library
docker exec ${CONTAINER_NAME} mkdir -p /var/lib/neo4j/temp
docker exec ${CONTAINER_NAME} chmod 777 /var/lib/neo4j/temp

# Set Java temp directory to avoid tmpfs noexec issues
docker exec -e "_JAVA_OPTIONS=-Djava.io.tmpdir=/var/lib/neo4j/temp" ${CONTAINER_NAME} neo4j-admin load \
    --database=${DATABASE_NAME} \
    --from=/var/lib/neo4j/data/dumps/import.dump \
    --force

echo "Step 6: Fixing database ownership..."
docker exec ${CONTAINER_NAME} bash -c "chown -R neo4j:neo4j /var/lib/neo4j/data/databases/${DATABASE_NAME}/ 2>/dev/null || true"
docker exec ${CONTAINER_NAME} bash -c "chown -R neo4j:neo4j /var/lib/neo4j/data/transactions/${DATABASE_NAME}/ 2>/dev/null || true"

echo "Step 7: Creating the database..."
docker exec ${CONTAINER_NAME} cypher-shell -u neo4j -p "${INITIAL_PASSWORD}" "CREATE DATABASE ${DATABASE_NAME};" 2>/dev/null || \
docker exec ${CONTAINER_NAME} cypher-shell -u neo4j -p "${YAML_PASSWORD}" "CREATE DATABASE ${DATABASE_NAME};"
sleep 10

echo "Step 8: Starting the database..."
docker exec ${CONTAINER_NAME} cypher-shell -u neo4j -p "${INITIAL_PASSWORD}" -d system "START DATABASE ${DATABASE_NAME};" 2>/dev/null || \
docker exec ${CONTAINER_NAME} cypher-shell -u neo4j -p "${YAML_PASSWORD}" -d system "START DATABASE ${DATABASE_NAME};"
sleep 5

echo "Step 9: Setting up users and passwords..."
if [ -f "$PROJECT_DIR/database/scripts/create_readonly_users.sh" ]; then
    bash "$PROJECT_DIR/database/scripts/create_readonly_users.sh"
else
    echo "Warning: create_readonly_users.sh not found, skipping user creation"
fi

echo "Step 10: Setting database to read-only mode..."
docker exec ${CONTAINER_NAME} cypher-shell -u neo4j -p "${YAML_PASSWORD}" -d system "ALTER DATABASE ${DATABASE_NAME} SET ACCESS READ ONLY;"

echo "Step 11: Restarting database in read-only mode..."
docker exec ${CONTAINER_NAME} cypher-shell -u neo4j -p "${YAML_PASSWORD}" -d system "STOP DATABASE ${DATABASE_NAME};" 2>/dev/null || true
sleep 3

# Ensure database starts successfully
for i in {1..3}; do
    docker exec ${CONTAINER_NAME} cypher-shell -u neo4j -p "${YAML_PASSWORD}" -d system "START DATABASE ${DATABASE_NAME};" 2>/dev/null
    sleep 5
    
    # Check if database is online
    if docker exec ${CONTAINER_NAME} cypher-shell -u neo4j -p "${YAML_PASSWORD}" -d system "SHOW DATABASE ${DATABASE_NAME};" 2>/dev/null | grep -q "online.*online"; then
        echo "✓ Database started successfully"
        break
    elif [ $i -eq 3 ]; then
        echo "⚠ Warning: Database may not be fully started"
    else
        echo "  Retrying start (attempt $((i+1))/3)..."
    fi
done

echo "Step 12: Cleaning up temporary file..."
docker exec ${CONTAINER_NAME} rm -f /var/lib/neo4j/data/dumps/import.dump

echo "Step 13: Verifying import..."
if docker exec ${CONTAINER_NAME} cypher-shell -u neo4j -p "${YAML_PASSWORD}" -d ${DATABASE_NAME} "MATCH (n) RETURN COUNT(n) as count LIMIT 1;" 2>/dev/null | grep -q "count"; then
    NODE_COUNT=$(docker exec ${CONTAINER_NAME} cypher-shell -u neo4j -p "${YAML_PASSWORD}" -d ${DATABASE_NAME} "MATCH (n) RETURN COUNT(n) as count;" 2>/dev/null | grep -E "^[0-9]+" | head -1)
    echo "✓ Database is accessible with $NODE_COUNT nodes"
    
    # Test reader access
    if docker exec ${CONTAINER_NAME} cypher-shell -u reader -p ReadOnly -d ${DATABASE_NAME} "MATCH (n) RETURN COUNT(n) as count LIMIT 1;" &>/dev/null; then
        echo "✓ Reader user can access database"
    else
        echo "⚠ Warning: Reader user cannot access database"
    fi
else
    echo "⚠ Warning: Could not verify database accessibility"
fi

echo ""
echo "========================================="
echo "Import completed successfully!"
echo "========================================="
echo ""
echo "Access URLs:"
echo "  HTTP:  http://localhost:7474"
echo "  HTTPS: https://torchcell-database.ncsa.illinois.edu:7473"
echo "  Bolt:  bolt+s://torchcell-database.ncsa.illinois.edu:7687"
echo ""
echo "Access credentials:"
echo "  Admin: neo4j / [from config file]"
echo "  Read-only: reader / ReadOnly"
echo ""
echo "Database: ${DATABASE_NAME}"
echo "Status: READ-ONLY"