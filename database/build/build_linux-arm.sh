#!/bin/bash -c
cd /Users/michaelvolk/Documents/projects/torchcell

echo "Starting build_linux-arm"
open -a Docker

sleep 3
docker login

# if tc-neo4j exists then remove it.
docker stop tc-neo4j
docker rm -f tc-neo4j

# Run the container
echo "Running container..."
# docker run -d --name tc-neo4j -p 7474:7474 -p 7687:7687 -v $(pwd)/database/biocypher-out:/database/biocypher-out -v $(pwd)/torchcell:/torchcell -v $(pwd)/data:/data -e NEO4J_AUTH=neo4j/torchcell michaelvolk/tc-neo4j:latest

# docker run --env=NEO4J_ACCEPT_LICENSE_AGREEMENT=yes -d --name tc-neo4j -p 7474:7474 -p 7687:7687 -v $(pwd)/database/biocypher-out:/database/biocypher-out -v $(pwd)/torchcell:/torchcell -v $(pwd)/data:/torchcell_data -v $(pwd)/database/data:/var/lib/neo4j/data -e NEO4J_AUTH=neo4j/torchcell michaelvolk/tc-neo4j:latest

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


# # Conda activate torchcell here since we are using the local library for the db writing.
# eval "$(conda shell.bash hook)"
# # conda init
# # source /Users/michaelvolk/miniconda3/etc/profile.d/conda.sh
# conda activate torchcell
# # conda activate /Users/michaelvolk/opt/miniconda3/envs/torchcell
# conda env list

# bash_script_path=$(python torchcell/knowledge_graphs/create_scerevisiae_kg_small.py)

docker start tc-neo4j
# TODO check if this software update works?
docker exec tc-neo4j python -m pip uninstall torchcell -y
docker exec tc-neo4j python -m pip install git+https://github.com/Mjvolk3/torchcell.git@main
docker exec tc-neo4j python -m pip uninstall biocypher -y
docker exec tc-neo4j python -m pip install git+https://github.com/Mjvolk3/biocypher@main

# docker exec -it tc-neo4j /bin/bash -c "chmod +x $bash_script_path"
# docker exec -it tc-neo4j /bin/bash -c "$bash_script_path"
echo "----------------NOW_BUILDING_GRAPHS---------------------"
# Execute the Python script inside the Docker container and capture the output
# bash_script_path=$(docker exec -it tc-neo4j python -m torchcell.knowledge_graphs.create_scerevisiae_kg_small)

docker exec tc-neo4j python -m torchcell.knowledge_graphs.create_scerevisiae_kg_small
bash_script_path_cleaned=$(docker exec tc-neo4j cat biocypher_file_name.txt)

# This only works if the stdout is completely clean
# bash_script_path_cleaned=$(docker exec tc-neo4j python -m torchcell.knowledge_graphs.create_scerevisiae_kg_small)

# echo "bash_script_path: $bash_script_path"
# Remove any unwanted characters (e.g., Docker exec command may include newline characters)
# bash_script_path_cleaned=$(echo "${bash_script_path}" | tr -d '\r' | tr -d '\n')

# echo "bash_script_path_cleaned: $bash_script_path_cleaned"

# Use the cleaned path in subsequent Docker exec commands
# For example, setting execute permission on the bash script
docker exec tc-neo4j /bin/bash -c "chmod +x ${bash_script_path_cleaned}"

# Execute the bash script
docker exec tc-neo4j /bin/bash -c "${bash_script_path_cleaned}"

# echo "Stopping container..."
# docker stop tc-neo4j

# echo "Removing container..."
# docker rm $container_id

# Extract the directory path without the filename, then protect the dir with no write permissions
dir_path=$(dirname "${bash_script_path_cleaned}")
docker exec tc-neo4j /bin/bash -c "chmod a-w '${dir_path}'"

echo "Build and run process completed."

# database is already started so don't need to work about starting. Apparently this is due to entry point in Dockerfile.tc-neo4j but it doesn't seem to work this way with apptainer. Apptainer might not see entry point?
