#!/bin/bash

# Activate Conda environment
source ~/.bashrc
conda activate myenv

# Uninstall and reinstall your packages
python -m pip uninstall torchcell -y
python -m pip install git+https://github.com/Mjvolk3/torchcell.git@main

python -m pip uninstall biocypher -y
python -m pip install git+https://github.com/Mjvolk3/biocypher@main

# TODO Should be moved into requirements.txt then removed after image build.
python -m pip install git+https://github.com/oxpig/CaLM@main

# Echo a message
echo "----------------NOW_BUILDING_GRAPHS----------------"

# Run your Python script and follow-up commands
# TODO for now we should delete  biocypher_file_name.txt prior to running adapters to eliminate confusion of loading previous graph.
rm -f biocypher_file_name.txt
python -m torchcell.knowledge_graphs.create_scerevisiae_kg
bash_script_path=$(cat biocypher_file_name.txt)
cd /var/lib/neo4j
chmod +x "${bash_script_path}"
/bin/bash -c "${bash_script_path}"

echo "Build and run process completed."
