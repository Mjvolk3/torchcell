#!/bin/bash

wandb_sync_agent_dirs() {
    # Exit on error, undefined vars, and pipe failures
    set -euo pipefail

    # Check if arguments were provided
    if [ $# -eq 0 ]; then
        echo "Error: No directories provided"
        echo "Usage: wandb_sync_agent_dirs dir1 dir2 ..."
        return 1
    fi  # Changed from } to fi

    # Source conda functions for non-interactive shells
    CONDA_BASE=$(conda info --base)
    if [ ! -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
        echo "Error: conda.sh not found at ${CONDA_BASE}/etc/profile.d/conda.sh"
        return 1
    fi  # Changed from } to fi

    source "${CONDA_BASE}/etc/profile.d/conda.sh"

    # Activate conda environment safely
    if ! conda activate torchcell 2>/dev/null; then
        echo "Error: Failed to activate torchcell environment"
        return 1
    fi  # Changed from } to fi

    # Export ADDR2LINE to prevent unbound variable warning
    export ADDR2LINE=""

    # Process each directory
    for exp_dir in "$@"; do
        wandb_dir="${exp_dir}/wandb"
        if [[ -d "${wandb_dir}" ]]; then
            echo "Syncing runs in ${wandb_dir}..."
            (
                cd "${wandb_dir}" || exit 1
                for d in offline-run-*/; do
                    if [[ -d "$d" ]]; then
                        echo "Syncing ${d}..."
                        wandb sync "${d}"
                    fi
                done
            )
        else
            echo "Directory ${wandb_dir} not found, skipping..."
        fi
    done
}

# Make the function available for export
export -f wandb_sync_agent_dirs

# wandb_sync_agent_dirs \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1535046_bf58cfe7e8d8117382e460b3af284ee441c8764da79870420147194156d98c28" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1535043_d030fb268eda9c675d05f82057f09f3bf35162e85406b541dce6c4e2cc612a1c" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1535042_dc5ed3baae554400cf6d7967184fd1c4db70ecd7388e0e2ce221769bc5b8cc5f" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1535041_41e4a1765b750b7d71e74e5d88db88d156b3225313296cd488cb076ecb571140" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1535040_9a673275ed83899c72946791093b79a20f398bfa86858597d0e087d13f854909"