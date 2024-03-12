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
bash_script_path=$(python -m torchcell.knowledge_graphs.create_scerevisiae_kg_small)
cd /var/lib/neo4j
chmod +x "${bash_script_path}"
/bin/bash -c "${bash_script_path}"
