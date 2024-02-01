#!/bin/bash
#SBATCH --mem=32g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32    # <- match to OMP_NUM_THREADS
#SBATCH --partition=cpu
#SBATCH --account=bbub-delta-cpu
#SBATCH --job-name=TC
#SBATCH --time=48:00:00
#SBATCH --constraint="projects&scratch"
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type="END"
#SBATCH --output=/projects/bbub/mjvolk3/torchcell/database/slurm/output/%x_%j.out
#SBATCH --error=/projects/bbub/mjvolk3/torchcell/database/slurm/error/%x_%j.out


# mem 128g
# cpu 128
module reset
source ~/.bashrc
cd /projects/bbub/mjvolk3/torchcell
conda activate /projects/bbub/miniconda3/envs/torchcell

echo "job is starting on `hostname`"

# Startup database
# apptainer run --writable-tmpfs   -B $(pwd)/neo4j/data:/data   -B $(pwd)/neo4j/logs:/logs   -B $(pwd)/neo4j/import:/var/lib/neo4j/import   -B $(pwd)/neo4j/plugins:/plugins   neo4j_4.4.30_community.sif

bash_script_path=$(python /projects/bbub/mjvolk3/torchcell/torchcell/knowledge_graphs/create_scerevisiae_kg_small.py)

echo bash_script_path
# cd to the directory containing the Neo4j database
#cd /path/to/neo4j/directory

# Run the Neo4j admin import script
#bash "$bash_script_path"

# Optionally, start the Neo4j database to verify the import or for further use
#./bin/neo4j start

# Perform any necessary operations with the database

# Finally, stop the Neo4j database before ending the job
#./bin/neo4j stop

#echo "Database shutdown and job complete."