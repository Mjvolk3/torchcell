#!/bin/bash
#SBATCH -p main
#SBATCH --mem=100g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --job-name=profile-compare
#SBATCH --time=02:00:00
#SBATCH --output=/home/michaelvolk/Documents/projects/torchcell/experiments/006-kuzmin-tmi/slurm/output/%x_%j.out
#SBATCH --error=/home/michaelvolk/Documents/projects/torchcell/experiments/006-kuzmin-tmi/slurm/output/%x_%j.out

# Profiling comparison SLURM script
# Runs both Dango and HeteroCell models sequentially with profiling enabled
# Then compares the results to identify why HeteroCell is 6x slower

cd /home/michaelvolk/Documents/projects/torchcell || exit 1

# Use single GPU for clean profiling
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="/home/michaelvolk/Documents/projects/torchcell:$PYTHONPATH"
export PYTHONUNBUFFERED=1

# Check if using container
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
    echo "Phase 1: Profiling HeteroCell Model"
    echo "========================================="
    python hetero_cell_bipartite_dango_gi.py

    echo ""
    echo "========================================="
    echo "Phase 2: Profiling Dango Model"
    echo "========================================="
    python dango.py

    echo ""
    echo "========================================="
    echo "Phase 3: Comparing Profiles"
    echo "========================================="
    python compare_profiler_outputs.py --auto-find
    '
else
    echo "Running without container"

    source ~/miniconda3/bin/activate
    conda activate torchcell

    cd experiments/006-kuzmin-tmi/scripts

    echo "========================================="
    echo "Phase 1: Profiling HeteroCell Model"
    echo "========================================="
    python hetero_cell_bipartite_dango_gi.py

    echo ""
    echo "========================================="
    echo "Phase 2: Profiling Dango Model"
    echo "========================================="
    python dango.py

    echo ""
    echo "========================================="
    echo "Phase 3: Comparing Profiles"
    echo "========================================="
    python compare_profiler_outputs.py --auto-find
fi

echo ""
echo "Profiling complete! Check output for performance bottlenecks."
echo "Profile outputs saved to: /scratch/projects/torchcell/data/torchcell/experiments/006-kuzmin-tmi/profiler_output/"