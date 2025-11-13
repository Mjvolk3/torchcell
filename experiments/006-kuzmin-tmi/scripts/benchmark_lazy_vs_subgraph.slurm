#!/bin/bash
#SBATCH -p main
#SBATCH --mem=100g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --job-name=bench-lazy-vs-subgraph
#SBATCH --time=02:00:00
#SBATCH --output=/home/michaelvolk/Documents/projects/torchcell/experiments/006-kuzmin-tmi/slurm/output/%x_%j.out
#SBATCH --error=/home/michaelvolk/Documents/projects/torchcell/experiments/006-kuzmin-tmi/slurm/output/%x_%j.out

# Benchmark LazySubgraphRepresentation vs SubgraphRepresentation
# Usage: sbatch benchmark_lazy_vs_subgraph.slurm

cd /home/michaelvolk/Documents/projects/torchcell || exit 1

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="/home/michaelvolk/Documents/projects/torchcell:$PYTHONPATH"
export PYTHONUNBUFFERED=1

if [ -f /home/michaelvolk/Documents/projects/torchcell/rockylinux_9.sif ]; then
    echo "Using container for benchmarking"

    apptainer exec --nv \
        --bind /scratch:/scratch \
        --bind /home/michaelvolk/Documents/projects/torchcell/.env:/home/michaelvolk/Documents/projects/torchcell/.env \
        --env PYTHONUNBUFFERED=1 \
        --env TORCHCELL_DEBUG_TIMING=1 \
        /home/michaelvolk/Documents/projects/torchcell/rockylinux_9.sif bash -lc '

        source ~/miniconda3/bin/activate
        conda activate torchcell

        cd /home/michaelvolk/Documents/projects/torchcell/experiments/006-kuzmin-tmi/scripts

        echo "========================================="
        echo "Benchmarking LazySubgraphRepresentation vs SubgraphRepresentation"
        echo "========================================="
        python benchmark_lazy_vs_subgraph.py
        '
else
    echo "Running without container"

    source ~/miniconda3/bin/activate
    conda activate torchcell

    cd experiments/006-kuzmin-tmi/scripts

    echo "========================================="
    echo "Benchmarking LazySubgraphRepresentation vs SubgraphRepresentation"
    echo "========================================="
    export TORCHCELL_DEBUG_TIMING=1
    echo "TORCHCELL_DEBUG_TIMING=${TORCHCELL_DEBUG_TIMING}"
    python benchmark_lazy_vs_subgraph.py
fi

echo ""
echo "Benchmark complete!"
echo "Check output above for results."
