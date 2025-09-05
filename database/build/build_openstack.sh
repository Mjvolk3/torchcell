#!/bin/bash

PROJECT_DIR="/home/rocky/projects/torchcell"
cd "$PROJECT_DIR" || exit 1

# Initial password for container startup
INITIAL_PASSWORD="torchcell"
# Read final password from YAML config (not committed to git)
CONFIG_FILE="$PROJECT_DIR/biocypher/config/PRODUCTION_linux-amd_biocypher_config.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found at $CONFIG_FILE"
    exit 1
fi
YAML_PASSWORD=$(grep "password:" "$CONFIG_FILE" | tail -1 | awk '{print $2}')

# Validate password was extracted
if [ -z "$YAML_PASSWORD" ]; then
    echo "Error: Could not extract password from $CONFIG_FILE"
    exit 1
fi
echo "Password successfully read from config file"

# Use home directory for data (need to manage disk space carefully)
DATA_DIR="$HOME/neo4j-data"

echo "Using local storage at: $DATA_DIR"

# Create directories locally
mkdir -p $DATA_DIR/{biocypher-out,torchcell,sgd,neo4j-data,logs}
chmod -R 777 $DATA_DIR

# Check for SSL certificates
CERT_DIR="$PROJECT_DIR/database/certificates/https"
if [ ! -f "$CERT_DIR/private.key" ] || [ ! -f "$CERT_DIR/public.crt" ]; then
    echo "No SSL certificates found."
    echo "For production, run: sudo bash $PROJECT_DIR/database/scripts/setup_letsencrypt.sh"
    echo "For development, generating self-signed certificates..."
    bash "$PROJECT_DIR/database/scripts/generate_ssl_certificates.sh"
elif [ -L "$CERT_DIR/private.key" ]; then
    echo "Using Let's Encrypt certificates"
else
    echo "Using existing SSL certificates"
fi

# Clean up existing container
docker stop tc-neo4j 2>/dev/null || true
docker rm -f tc-neo4j 2>/dev/null || true

# Start container with LOCAL storage
echo "Starting tc-neo4j container..."
docker run \
    --cpus=14 \
    --memory=56g \
    --tmpfs /tmp:size=10G \
    --env=NEO4J_ACCEPT_LICENSE_AGREEMENT=yes \
    -d --name tc-neo4j \
    -p 7474:7474 -p 7473:7473 -p 7687:7687 \
    -v "$DATA_DIR/biocypher-out:/var/lib/neo4j/biocypher-out" \
    -v "$DATA_DIR/torchcell:/var/lib/neo4j/data/torchcell" \
    -v "$DATA_DIR/sgd:/var/lib/neo4j/data/sgd" \
    -v "$DATA_DIR/neo4j-data:/var/lib/neo4j/data" \
    -v "$(pwd)/database/.env:/.env:ro" \
    -v "$(pwd)/biocypher:/var/lib/neo4j/biocypher" \
    -v "$(pwd)/database/conf:/var/lib/neo4j/conf" \
    -v "$(pwd)/database/certificates:/var/lib/neo4j/certificates" \
    --tmpfs /logs:size=1G \
    -v "$(pwd)/database/plugins:/plugins" \
    -e NEO4J_AUTH=neo4j/torchcell \
    -e NEO4J_dbms_read__only=false \
    -e NEO4J_apoc_export_file_enabled=true \
    -e NEO4J_apoc_import_file_enabled=true \
    -e NEO4J_apoc_import_file_use__neo4j__config=true \
    -e NEO4J_dbms_security_procedures_unrestricted=apoc.\* \
    michaelvolk/tc-neo4j:latest

sleep 30

# Check if container is running
if ! docker ps | grep -q tc-neo4j; then
    echo "Container failed to start. Checking logs..."
    docker logs --tail 50 tc-neo4j
    exit 1
fi

echo "Container started successfully!"
docker exec tc-neo4j df -h /

# Rest of script remains the same...
echo "Installing packages..."
docker exec tc-neo4j bash -c "\
source /root/.bashrc && \
conda activate myenv && \
python -m pip install --upgrade pip && \
python -m pip uninstall -y torchcell && \
python -m pip install git+https://github.com/Mjvolk3/torchcell.git@main && \
python -m pip install --force-reinstall --no-cache-dir torch_scatter -f https://data.pyg.org/whl/torch-2.8.0+cpu.html && \
python -m pip uninstall -y biocypher && \
python -m pip install git+https://github.com/Mjvolk3/biocypher@main && \
python -m pip install git+https://github.com/oxpig/CaLM@main && \
python -m pip install --no-cache-dir hypernetx fastjsonschema"

# WandB login
echo "Logging into WandB..."
docker exec tc-neo4j bash -c '[ -f /.env ] && source /.env && wandb login "$WANDB_API_KEY" || echo ".env not found"'

# Build graphs
echo "Building knowledge graphs..."
docker exec tc-neo4j bash -c "\
source /root/.bashrc && \
conda activate myenv && \
cd /var/lib/neo4j && \
python -m torchcell.knowledge_graphs.create_kg --config-name=smf_costanzo2016_kg"

bash_script_path_cleaned=$(docker exec tc-neo4j cat /var/lib/neo4j/biocypher_file_name.txt 2>/dev/null)

if [ -z "$bash_script_path_cleaned" ]; then
    echo "Failed to retrieve the script path."
    exit 1
fi

docker exec tc-neo4j bash -c "chmod +x ${bash_script_path_cleaned}"

# Drop the existing database if it exists to avoid "database in use" error
echo "Dropping existing torchcell database if it exists..."
docker exec tc-neo4j cypher-shell -u neo4j -p "$INITIAL_PASSWORD" -d system "DROP DATABASE torchcell IF EXISTS;" 2>/dev/null || \
docker exec tc-neo4j cypher-shell -u neo4j -p "$YAML_PASSWORD" -d system "DROP DATABASE torchcell IF EXISTS;" 2>/dev/null || true
sleep 5

# Run the import - this creates the database files
echo "Running import script..."
docker exec tc-neo4j bash -c "cd /var/lib/neo4j && ${bash_script_path_cleaned}"

# CRITICAL: Fix ownership of ALL imported files (import runs as root)
echo "Fixing database and transaction log ownership..."
# Fix database files
docker exec tc-neo4j bash -c "chown -R neo4j:neo4j /var/lib/neo4j/data/databases/torchcell/ 2>/dev/null || true"
docker exec tc-neo4j bash -c "chown -R neo4j:neo4j /data/databases/torchcell/ 2>/dev/null || true"
# Fix transaction logs (CRITICAL - often missed)
docker exec tc-neo4j bash -c "chown -R neo4j:neo4j /var/lib/neo4j/data/transactions/torchcell/ 2>/dev/null || true"
docker exec tc-neo4j bash -c "chown -R neo4j:neo4j /data/transactions/torchcell/ 2>/dev/null || true"

# Create the torchcell database after import
echo "Creating torchcell database..."
docker exec tc-neo4j cypher-shell -u neo4j -p "$INITIAL_PASSWORD" "CREATE DATABASE torchcell IF NOT EXISTS;" 2>/dev/null || \
docker exec tc-neo4j cypher-shell -u neo4j -p "$YAML_PASSWORD" "CREATE DATABASE torchcell IF NOT EXISTS;"

# Wait for database to initialize
sleep 10

# Start the database (required before setting to READ ONLY)
echo "Starting torchcell database..."
docker exec tc-neo4j cypher-shell -u neo4j -p "$INITIAL_PASSWORD" -d system "START DATABASE torchcell;" 2>/dev/null || \
docker exec tc-neo4j cypher-shell -u neo4j -p "$YAML_PASSWORD" -d system "START DATABASE torchcell;"
sleep 5

# Update neo4j password and create users
echo "Setting up users and passwords..."
if [ -f "$PROJECT_DIR/database/scripts/create_readonly_users.sh" ]; then
    bash "$PROJECT_DIR/database/scripts/create_readonly_users.sh"
else
    echo "Warning: create_readonly_users.sh not found, skipping user creation"
fi

# Now set database to read-only (must be done with the updated password)
echo "Setting database to read-only mode..."
docker exec tc-neo4j cypher-shell -u neo4j -p "$YAML_PASSWORD" -d system "ALTER DATABASE torchcell SET ACCESS READ ONLY;"

# Database must be restarted after setting to READ ONLY
echo "Restarting torchcell database in read-only mode..."
docker exec tc-neo4j cypher-shell -u neo4j -p "$YAML_PASSWORD" -d system "STOP DATABASE torchcell;" 2>/dev/null || true
sleep 3

# Ensure database starts successfully
echo "Starting database..."
for i in {1..3}; do
    docker exec tc-neo4j cypher-shell -u neo4j -p "$YAML_PASSWORD" -d system "START DATABASE torchcell;" 2>/dev/null
    sleep 5
    
    # Check if database is online
    if docker exec tc-neo4j cypher-shell -u neo4j -p "$YAML_PASSWORD" -d system "SHOW DATABASE torchcell;" 2>/dev/null | grep -q "online.*online"; then
        echo "✓ Database started successfully"
        break
    elif [ $i -eq 3 ]; then
        echo "⚠ Warning: Database may not be fully started"
    else
        echo "  Retrying start (attempt $((i+1))/3)..."
    fi
done

# Verify database is accessible
echo "Verifying database is accessible..."
if docker exec tc-neo4j cypher-shell -u neo4j -p "$YAML_PASSWORD" -d torchcell "MATCH (n) RETURN COUNT(n) as count LIMIT 1;" 2>/dev/null | grep -q "count"; then
    NODE_COUNT=$(docker exec tc-neo4j cypher-shell -u neo4j -p "$YAML_PASSWORD" -d torchcell "MATCH (n) RETURN COUNT(n) as count;" 2>/dev/null | grep -E "^[0-9]+" | head -1)
    echo "✓ Database is accessible with $NODE_COUNT nodes"
    
    # Test reader access too
    if docker exec tc-neo4j cypher-shell -u reader -p ReadOnly -d torchcell "MATCH (n) RETURN COUNT(n) as count LIMIT 1;" &>/dev/null; then
        echo "✓ Reader user can access database"
    else
        echo "⚠ Warning: Reader user cannot access database"
    fi
else
    echo "⚠ Warning: Could not verify database accessibility"
    echo "  Checking database status..."
    docker exec tc-neo4j cypher-shell -u neo4j -p "$YAML_PASSWORD" -d system "SHOW DATABASE torchcell;" 2>/dev/null | grep torchcell || true
fi

echo "Database build complete and set to read-only."
echo ""
echo "Access URLs:"
echo "  HTTP:  http://localhost:7474"
echo "  HTTPS: https://torchcell-database.ncsa.illinois.edu:7473"
echo "  Bolt:  bolt+s://torchcell-database.ncsa.illinois.edu:7687"
echo ""
echo "Access credentials:"
echo "  Admin: neo4j / ***************"
echo "  Read-only: reader / ReadOnly"

# Fix permissions so rocky can still access the files
echo ""
echo "Fixing file permissions for continued development..."
sudo chown -R rocky:neo4j "$PROJECT_DIR/biocypher/" 2>/dev/null || true
sudo chown -R rocky:neo4j "$PROJECT_DIR/database/conf/" 2>/dev/null || true
sudo chown -R rocky:neo4j "$PROJECT_DIR/database/certificates/" 2>/dev/null || true
echo "Permissions fixed - rocky can now access configuration and certificate files"