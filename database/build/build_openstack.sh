#!/bin/bash

PROJECT_DIR="/home/rocky/projects/torchcell"

cd "$PROJECT_DIR" || exit 1

# Docker login (if needed)
docker login

# Clean up existing container
docker stop tc-neo4j || true
docker rm -f tc-neo4j || true

# Start Docker container
echo "Starting tc-neo4j container..."
docker run \
    --cpus=10 \
    --memory=128g \
    --env=NEO4J_ACCEPT_LICENSE_AGREEMENT=yes \
    -d --name tc-neo4j \
    -p 7474:7474 -p 7687:7687 \
    -v "$(pwd)/database/biocypher-out:/var/lib/neo4j/biocypher-out" \
    -v "$(pwd)/data/torchcell:/var/lib/neo4j/data/torchcell" \
    -v "$(pwd)/data/sgd:/var/lib/neo4j/data/sgd" \
    -v "$(pwd)/database/data:/var/lib/neo4j/data" \
    -v "$(pwd)/database/.env:/.env" \
    -v "$(pwd)/biocypher:/var/lib/neo4j/biocypher" \
    -v "$(pwd)/database/conf:/var/lib/neo4j/conf" \
    -v "$(pwd)/database/logs:/logs" \
    -v "$(pwd)/database/plugins:/plugins" \
    -e NEO4J_AUTH=neo4j/torchcell \
    -e NEO4J_dbms_read__only=false \
    -e NEO4J_apoc_export_file_enabled=true \
    -e NEO4J_apoc_import_file_enabled=true \
    -e NEO4J_apoc_import_file_use__neo4j__config=true \
    -e NEO4J_dbms_security_procedures_unrestricted=apoc.\* \
    michaelvolk/tc-neo4j:latest

sleep 10

# Ensure container is running
if ! docker ps | grep -q tc-neo4j; then
    echo "Container tc-neo4j failed to start."
    exit 1
fi

# Update packages inside container
docker exec tc-neo4j python -m pip install --upgrade pip
docker exec tc-neo4j python -m pip uninstall torchcell -y
docker exec tc-neo4j python -m pip install git+https://github.com/Mjvolk3/torchcell.git@main
docker exec tc-neo4j python -m pip install --force-reinstall --no-cache torch_scatter -f https://data.pyg.org/whl/torch-2.4.0+cpu.html
docker exec tc-neo4j python -m pip uninstall biocypher -y
docker exec tc-neo4j python -m pip install git+https://github.com/Mjvolk3/biocypher@main
docker exec tc-neo4j python -m pip install git+https://github.com/oxpig/CaLM@main
docker exec tc-neo4j python -m pip install --no-cache-dir hypernetx fastjsonschema

# WandB login
echo "Logging into WandB..."
docker exec tc-neo4j bash -c 'source /.env && wandb login "$WANDB_API_KEY"'

# Build graphs
echo "Building knowledge graphs..."
docker exec tc-neo4j python -m torchcell.knowledge_graphs.create_kg --config-name=smf_costanzo2016_kg

bash_script_path_cleaned=$(docker exec tc-neo4j cat biocypher_file_name.txt)

if [ -z "$bash_script_path_cleaned" ]; then
    echo "Failed to retrieve the script path."
    exit 1
fi

docker exec tc-neo4j bash -c "chmod +x ${bash_script_path_cleaned}"
docker exec tc-neo4j bash -c "${bash_script_path_cleaned}"

# Secure directory permissions
dir_path=$(dirname "${bash_script_path_cleaned}")
docker exec tc-neo4j bash -c "chmod a-w '${dir_path}'"

echo "Database build complete."
