#!/bin/bash

# Check if an argument was provided
if [ "$#" -ne 1 ]; then
    read -p "Enter the argument for the command: " user_input
else
    user_input="$1"
fi

# Command to execute
command_to_run="rsync -avz -e 'ssh -l mjvolk3' mjvolk3@dt-login02.delta.ncsa.illinois.edu://scratch/bbub/mjvolk3/torchcell/database/biocypher-out/'${user_input}' /Users/michaelvolk/Documents/projects/torchcell/database/biocypher-out"

# Execute the command
eval $command_to_run
