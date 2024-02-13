#!/bin/bash -c
open -a Docker

sleep 10

docker login

echo "Building image..."
docker buildx build --platform linux/amd64,linux/arm64 -t michaelvolk/tc-neo4j:latest -f database/Dockerfile.tc-neo4j database --push

echo "Pulling latest image..."
docker pull michaelvolk/tc-neo4j:latest

######### 

#!/bin/bash -c
cd /Users/michaelvolk/Documents/projects/torchcell

echo "Starting build_linux-arm"
open -a Docker

sleep 3
docker login

# if tc-neo4j exists then remove it.
docker stop tc-neo4j
docker rm -f tc-neo4j

# Rebuild image
echo "Building image..."
docker buildx build --platform linux/amd64,linux/arm64 -t michaelvolk/tc-neo4j:latest -f database/Dockerfile.tc-neo4j database --push

echo "Pulling latest image..."
docker pull michaelvolk/tc-neo4j:latest

# Run the container
echo "Running container..."
container_id=$(docker run -d --name tc-neo4j -p 7474:7474 -p 7687:7687 -v $(pwd)/database/biocypher-out:/database/biocypher-out michaelvolk/tc-neo4j:latest)

# Change the Neo4j password to 'torchcell'
echo "Changing Neo4j password..."
docker exec -it tc-neo4j cypher-shell -u neo4j -p neo4j "ALTER USER neo4j SET PASSWORD 'torchcell';"

# Conda activate torchcell here since we are using the local library for the db writing.
eval "$(conda shell.bash hook)"
# conda init
# source /Users/michaelvolk/miniconda3/etc/profile.d/conda.sh  
conda activate torchcell
# conda activate /Users/michaelvolk/opt/miniconda3/envs/torchcell
conda env list

bash_script_path=$(python torchcell/knowledge_graphs/create_scerevisiae_kg_small.py)

docker exec -it tc-neo4j /bin/bash -c "chmod +x $bash_script_path"
docker exec -it tc-neo4j /bin/bash -c "$bash_script_path"

echo "Stopping container..."
docker stop $container_id

echo "Removing container..."
docker rm $container_id

echo "Build and run process completed."


# database is already started so don't need to work about starting. Apparently this is due to entry point in Dockerfile.tc-neo4j but it doesn't seem to work this way with apptainer. Apptainer might not see entry point?

