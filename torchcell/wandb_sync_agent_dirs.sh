#!/bin/bash
# torchcell/wandb_sync_agent_dirs.sh

# Function to sync all runs within wandb directories
wandb_sync_agent_dirs() {
    # Check if arguments were provided
    if [ $# -eq 0 ]; then
        echo "Error: No directories provided"
        echo "Usage: wandb_sync_agent_dirs dir1 dir2 ..."
        return 1
    fi
    
    # Try to activate conda environment if needed
    if command -v conda >/dev/null 2>&1; then
        CONDA_BASE=$(conda info --base 2>/dev/null)
        if [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
            source "${CONDA_BASE}/etc/profile.d/conda.sh"
            conda activate torchcell 2>/dev/null || echo "Note: Could not activate torchcell environment"
        fi
    fi
    
    total_dirs=0
    successful_syncs=0
    
    # Process each experiment directory
    for exp_dir in "$@"; do
        # Find the wandb directory
        if [ -d "${exp_dir}/wandb" ]; then
            wandb_dir="${exp_dir}/wandb"
        else
            wandb_dir="${exp_dir}"
        fi
        
        if [ -d "$wandb_dir" ]; then
            total_dirs=$((total_dirs + 1))
            echo "--------------------------------------------"
            echo "Processing experiment: $wandb_dir"
            
            # Use ls to list directories and process each one separately
            orig_dir=$(pwd)
            cd "$wandb_dir" || { echo "Error: Could not cd to $wandb_dir"; continue; }
            
            # Process run directories first
            for run in $(ls -d run-* 2>/dev/null || true); do
                if [ -d "$run" ]; then
                    echo "Syncing online run: $run"
                    wandb sync --include-online --include-offline --include-synced "$run"
                    if [ $? -eq 0 ]; then
                        successful_syncs=$((successful_syncs + 1))
                    fi
                fi
            done
            
            # Then process offline runs
            for run in $(ls -d offline-run-* 2>/dev/null || true); do
                if [ -d "$run" ]; then
                    echo "Syncing offline run: $run"
                    wandb sync --include-online --include-offline --include-synced "$run"
                    if [ $? -eq 0 ]; then
                        successful_syncs=$((successful_syncs + 1))
                    fi
                fi
            done
            
            # Return to original directory
            cd "$orig_dir"
        else
            echo "Directory not found: $wandb_dir"
        fi
    done
    
    echo "--------------------------------------------"
    echo "Finished processing $total_dirs experiment directories"
    echo "Successfully synced $successful_syncs runs"
    echo "--------------------------------------------"
}