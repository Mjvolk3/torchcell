#!/bin/bash
#SBATCH -p main
#SBATCH --mem=100g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --job-name=profile-opt
#SBATCH --time=01:00:00
#SBATCH --output=/home/michaelvolk/Documents/projects/torchcell/experiments/006-kuzmin-tmi/slurm/output/%x_%j.out
#SBATCH --error=/home/michaelvolk/Documents/projects/torchcell/experiments/006-kuzmin-tmi/slurm/output/%x_%j.out

# Profile HeteroCell model with stable config
# Usage: sbatch profile_single_model.slurm

cd /home/michaelvolk/Documents/projects/torchcell || exit 1

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="/home/michaelvolk/Documents/projects/torchcell:$PYTHONPATH"
export PYTHONUNBUFFERED=1

if [ -f /home/michaelvolk/Documents/projects/torchcell/rockylinux_9.sif ]; then
    echo "Using container for profiling"

    apptainer exec --nv \
        --bind /scratch:/scratch \
        --bind /home/michaelvolk/Documents/projects/torchcell/.env:/home/michaelvolk/Documents/projects/torchcell/.env \
        --env PYTHONUNBUFFERED=1 \
        /home/michaelvolk/Documents/projects/torchcell/rockylinux_9.sif bash -lc '

        source ~/miniconda3/bin/activate
        conda activate torchcell

        cd /home/michaelvolk/Documents/projects/torchcell/experiments/006-kuzmin-tmi/scripts

        echo "========================================="
        echo "Profiling HeteroCell with Stable Config"
        echo "========================================="
        python hetero_cell_bipartite_dango_gi.py --config-name profiling_stable_config
        '
else
    echo "Running without container"

    source ~/miniconda3/bin/activate
    conda activate torchcell

    cd experiments/006-kuzmin-tmi/scripts

    echo "========================================="
    echo "Profiling HeteroCell with Stable Config"
    echo "========================================="
    python hetero_cell_bipartite_dango_gi.py --config-name profiling_stable_config
fi

echo ""
echo "Profiling complete!"
echo "Profile outputs saved to: /scratch/projects/torchcell/data/torchcell/experiments/006-kuzmin-tmi/profiler_output/"
