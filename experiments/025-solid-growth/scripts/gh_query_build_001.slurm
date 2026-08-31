#!/bin/bash
#SBATCH --job-name=025-query-build
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=25
#SBATCH --mem=100G
#SBATCH --time=3-00:00:00
#SBATCH --output=/home/michaelvolk/Documents/projects/torchcell/experiments/025-solid-growth/slurm/output/%j_%x.out
#SBATCH --error=/home/michaelvolk/Documents/projects/torchcell/experiments/025-solid-growth/slurm/output/%j_%x.out
#SBATCH --no-requeue

# Build the 025 all-solid-growth dataset from the served graph, under slurm so the
# work is scheduled against the node's other jobs and memory-capped so a whole-dataset
# materialization kills THIS JOB, not the machine. That cap has now caught two real
# ones: the driver node-cache leak (32 G cap was fine for the fixed fetch -- 1.9 G RSS
# across 43.8M records) and the reference-index list materialization (OOM'd job 1559
# six minutes after an 18 h fetch; fixed streaming). 100 G gives the remaining
# stages -- conversion, dedup maps, aggregation, the 43.8M-row label DataFrame --
# headroom that is a guess, not a measurement; the cap exists precisely because it
# is a guess. MaxMemPerCPU=4100 makes 100 G imply a 25-CPU floor, matching the
# request.
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
