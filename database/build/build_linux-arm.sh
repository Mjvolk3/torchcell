#!/bin/bash

cd /Users/michaelvolk/Documents/projects/torchcell || exit

echo "Starting build_linux-arm"
open -a Docker

sleep 3
docker login

# If tc-neo4j exists then remove it.
docker stop tc-neo4j || true
docker rm -f tc-neo4j || true

# Run the container
echo "Running container..."
docker run --cpus=10 \
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

# Allow some time for the container to start
sleep 10

# Verify the container is running
docker ps | grep tc-neo4j
if [ $? -ne 0 ]; then
    echo "Container tc-neo4j failed to start."
    exit 1
fi


docker exec tc-neo4j python -m pip install --upgrade pip
# Install required packages inside the container
docker exec tc-neo4j python -m pip uninstall torchcell -y

docker exec tc-neo4j python -m pip install git+https://github.com/Mjvolk3/torchcell.git@main
docker exec tc-neo4j python -m pip install --force-reinstall --no-cache torch_scatter -f https://data.pyg.org/whl/torch-2.4.0+cpu.html
docker exec tc-neo4j python -m pip uninstall biocypher -y
docker exec tc-neo4j python -m pip install git+https://github.com/Mjvolk3/biocypher@main
docker exec tc-neo4j python -m pip install git+https://github.com/oxpig/CaLM@main
# Can move some to reqs.
docker exec tc-neo4j python -m pip install --no-cache-dir hypernetx
docker exec tc-neo4j python -m pip install --no-cache-dir fastjsonschema 


echo "----------------NOW_BUILDING_GRAPHS---------------------"

echo "Logging in to wandb..."
docker exec tc-neo4j bash -c 'source /.env && wandb login $WANDB_API_KEY'

# Execute the Python script inside the Docker container and capture the output
#Works
#docker exec tc-neo4j python -m torchcell.knowledge_graphs.create_kg --config-name=gene_essentiality_sgd

docker exec tc-neo4j python -m torchcell.knowledge_graphs.create_kg --config-name=smf_costanzo2016_kg

# Capture the path from the script output
bash_script_path_cleaned=$(docker exec tc-neo4j cat biocypher_file_name.txt)

# Ensure the file exists and is not empty
if [ -z "$bash_script_path_cleaned" ]; then
    echo "Failed to retrieve the script path."
    exit 1
fi

# Use the cleaned path in subsequent Docker exec commands
docker exec tc-neo4j /bin/bash -c "chmod +x ${bash_script_path_cleaned}"
docker exec tc-neo4j /bin/bash -c "${bash_script_path_cleaned}"

# Extract the directory path without the filename, then protect the dir with no write permissions
dir_path=$(dirname "${bash_script_path_cleaned}")
docker exec tc-neo4j /bin/bash -c "chmod a-w '${dir_path}'"

# Not necessary on Mac - fails to create
#docker exec tc-neo4j bash -c 'source /.env && cypher-shell -u "$NEO4J_USER" -p "$NEO4J_PASSWORD" "CREATE DATABASE torchcell;"'

echo "Build and run process completed."

docker stop tc-neo4j
docker start tc-neo4j
