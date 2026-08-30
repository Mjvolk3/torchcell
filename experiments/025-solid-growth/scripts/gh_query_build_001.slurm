#!/bin/bash
#SBATCH --job-name=025-query-build
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH --output=/home/michaelvolk/Documents/projects/torchcell/experiments/025-solid-growth/slurm/output/%j_%x.out
#SBATCH --error=/home/michaelvolk/Documents/projects/torchcell/experiments/025-solid-growth/slurm/output/%j_%x.out
#SBATCH --no-requeue

# Build the 025 all-solid-growth dataset from the served graph, under slurm so the
# work is scheduled against the node's other jobs and memory-capped so a regression
# of the driver node-cache leak kills THIS JOB, not the machine. Post-fix heap is
# flat (measured 3 B/record retained); 32 G is pure ceiling. The scheduler's
# MaxMemPerCPU=4100 makes 32 G imply an 8-CPU floor, matching the request.
#
# The dataset root symlinks to /db (the SN850X): the pipeline materializes five
# near-equal stage copies (raw/conversion/dedup/aggregation/processed), and the
# projected total does not fit /scratch. A dataset build is regenerable, which is
# the /db rule.
#
#   sbatch experiments/025-solid-growth/scripts/gh_query_build_001.slurm

echo "Running on $(hostname)"
source ~/.zshrc
conda activate ~/miniconda3/envs/torchcell

cd /home/michaelvolk/Documents/projects/torchcell

export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="torchcell"

python experiments/025-solid-growth/scripts/query.py
