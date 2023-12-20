#!/bin/bash -c
cd /usr/app/
# cp -r /src/* . #TEST
######
# Copy specific directories and files
cp -r /src/Dockerfile .
cp -r /src/README.md .
cp -r /src/biocypher-log .
cp -r /src/biocypher-out .
cp -r /src/collectri .
cp -r /src/config .
# cp -r /src/create_knowledge_graph.py .
cp -r /src/tc_create_knowledge_graph.py .
cp -r /src/docker .
cp -r /src/docker-compose-chatgse.yml .
cp -r /src/docker-compose.yml .
cp -r /src/docker-variables.env .
cp -r /src/env .
cp -r /src/scripts .
cp -r /src/env .
#####
cp config/biocypher_docker_config.yaml config/biocypher_config.yaml

# Activate Conda
source /miniconda/etc/profile.d/conda.sh

# Check if Conda environment exists and update or create accordingly
if conda env list | grep -q 'tc-graph'; then
    echo "Updating existing Conda environment: tc-graph"
    conda env update -f env/tc-graph-docker.yaml -n tc-graph
else
    echo "Creating new Conda environment: tc-graph"
    conda env create -f env/tc-graph-docker.yaml
fi

conda activate tc-graph

# Run Python script
python3 tc_create_knowledge_graph.py

chmod -R 777 biocypher-log
