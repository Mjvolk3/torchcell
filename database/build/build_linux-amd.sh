#!/bin/bash

# Activate Conda environment
source ~/.bashrc
conda activate myenv

# Uninstall and reinstall your packages
python -m pip uninstall torchcell -y
python -m pip install git+https://github.com/Mjvolk3/torchcell.git@main

python -m pip uninstall biocypher -y
python -m pip install git+https://github.com/Mjvolk3/biocypher@main

# Echo a message
echo "----------------NOW_BUILDING_GRAPHS----------------"

# Run your Python script and follow-up commands
python -m torchcell.knowledge_graphs.create_scerevisiae_kg
bash_script_path=$(cat biocypher_file_name.txt)
cd /var/lib/neo4j
chmod +x "${bash_script_path}"
/bin/bash -c "${bash_script_path}"

dir_path=$(dirname "${bash_script_path_cleaned}")
docker exec tc-neo4j /bin/bash -c "chmod a-w '${dir_path}'"

echo "Build and run process completed."
