#!/bin/bash

# Base directory containing the experiment directories
base_dir="/Users/michaelvolk/Documents/projects/torchcell/wandb-experiments"

# Iterate over each directory in the base directory
for dir in "$base_dir"/*; do
  if [ -d "$dir" ]; then
    # Check if the 'wandb' subdirectory exists
    if [ -d "$dir/wandb" ]; then
      # Sync the 'wandb' subdirectory
      wandb sync "$dir/wandb"
    else
      echo "No 'wandb' subdirectory found in $dir"
    fi
  fi
done