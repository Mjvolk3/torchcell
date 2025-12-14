#!/bin/bash

# Neo4j Database Export Script
# Exports the torchcell database using neo4j-admin dump

set -e

# Configuration
CONTAINER_NAME="tc-neo4j"
DATABASE_NAME="torchcell"
EXPORT_DIR="/scratch/projects/torchcell-scratch/database/export"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPORT_FILE="${DATABASE_NAME}_${TIMESTAMP}.dump"

echo "========================================="
echo "Neo4j Database Export Script"
echo "========================================="
echo "Database: ${DATABASE_NAME}"
echo "Export to: ${EXPORT_DIR}/${EXPORT_FILE}"
echo "Timestamp: ${TIMESTAMP}"
echo ""

# Check if container is running
if ! docker ps -q -f name=${CONTAINER_NAME} > /dev/null; then
    echo "Error: Container ${CONTAINER_NAME} is not running"
    exit 1
fi

echo "Step 1: Stopping the database for consistent export..."
docker exec ${CONTAINER_NAME} bash -c "source /.env && cypher-shell -u \"\$NEO4J_USER\" -p \"\$NEO4J_PASSWORD\" \"STOP DATABASE ${DATABASE_NAME};\""

echo "Step 2: Creating dump of database..."
# Create dump inside container
docker exec ${CONTAINER_NAME} neo4j-admin dump \
    --database=${DATABASE_NAME} \
    --to=/tmp/${EXPORT_FILE}

echo "Step 3: Copying dump file to export directory..."
# Copy dump from container to host
docker cp ${CONTAINER_NAME}:/tmp/${EXPORT_FILE} ${EXPORT_DIR}/${EXPORT_FILE}

echo "Step 4: Cleaning up temporary file in container..."
docker exec ${CONTAINER_NAME} rm -f /tmp/${EXPORT_FILE}

echo "Step 5: Restarting the database..."
docker exec ${CONTAINER_NAME} bash -c "source /.env && cypher-shell -u \"\$NEO4J_USER\" -p \"\$NEO4J_PASSWORD\" \"START DATABASE ${DATABASE_NAME};\""

# Verify the export
if [ -f "${EXPORT_DIR}/${EXPORT_FILE}" ]; then
    FILE_SIZE=$(ls -lh ${EXPORT_DIR}/${EXPORT_FILE} | awk '{print $5}')
    echo ""
    echo "========================================="
    echo "Export completed successfully!"
    echo "File: ${EXPORT_DIR}/${EXPORT_FILE}"
    echo "Size: ${FILE_SIZE}"
    echo "========================================="
    
    # Create a latest symlink for easy access
    ln -sf ${EXPORT_FILE} ${EXPORT_DIR}/${DATABASE_NAME}_latest.dump
    echo "Created symlink: ${EXPORT_DIR}/${DATABASE_NAME}_latest.dump"
else
    echo "Error: Export file was not created"
    exit 1
fi

# Optional: List all exports
echo ""
echo "All database exports:"
ls -lht ${EXPORT_DIR}/*.dump | head -10