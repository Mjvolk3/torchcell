#!/bin/bash -c
cd "$PWD"

echo "Starting build_gilahyper-docker"
source .env

sleep 3
docker login

# if tc-neo4j exists then remove it.
docker stop tc-neo4j
docker rm -f tc-neo4j

# Run the container
echo "Running container..."

docker run --cpus=10 \
    --env=NEO4J_ACCEPT_LICENSE_AGREEMENT=yes \
    -d --name tc-neo4j \
    -p 7474:7474 -p 7687:7687 \
    -v $(pwd)/database/biocypher-out:/var/lib/neo4j/biocypher-out \
    -v $(pwd)/data/torchcell:/var/lib/neo4j/data/torchcell \
    -v $(pwd)/database/data:/var/lib/neo4j/data \
    -v $(pwd)/database/.env:/.env \
    -v $(pwd)/biocypher:/var/lib/neo4j/biocypher \
    -v $(pwd)/database/conf:/var/lib/neo4j/conf \
    -v $(pwd)/database/logs:/logs \
    -e NEO4J_AUTH=neo4j/torchcell \
    michaelvolk/tc-neo4j:latest

docker start tc-neo4j
# TODO check if this software update works?
docker exec tc-neo4j python -m pip uninstall torchcell -y
docker exec tc-neo4j python -m pip install git+https://github.com/Mjvolk3/torchcell.git@main
docker exec tc-neo4j python -m pip uninstall biocypher -y
docker exec tc-neo4j python -m pip install git+https://github.com/Mjvolk3/biocypher@main
# TODO Should be moved into requirements.txt then removed after image build.
docker exec tc-neo4j python -m pip install git+https://github.com/oxpig/CaLM@main
docker exec tc-neo4j python -m pip install memory-profiler

docker start tc-neo4j

# Add wandb login command
docker exec tc-neo4j wandb login $WANDB_API_KEY

echo "----------------NOW_BUILDING_GRAPHS---------------------"

docker exec tc-neo4j python -m torchcell.knowledge_graphs.create_scerevisiae_kg_small
# Is this a temp file in docker container? Yes I think so.
bash_script_path_cleaned=$(docker exec tc-neo4j cat biocypher_file_name.txt)

docker exec tc-neo4j /bin/bash -c "chmod +x ${bash_script_path_cleaned}"

# Execute the bash script
docker exec tc-neo4j /bin/bash -c "${bash_script_path_cleaned}"

# Extract the directory path without the filename, then protect the dir with no write permissions
dir_path=$(dirname "${bash_script_path_cleaned}")
docker exec tc-neo4j /bin/bash -c "chmod a-w '${dir_path}'"

echo "Build and run process completed."