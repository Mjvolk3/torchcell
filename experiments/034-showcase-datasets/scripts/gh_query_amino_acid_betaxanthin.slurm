#!/bin/bash
#SBATCH --job-name=034-showcase-amino-acid-betaxanthin
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=experiments/034-showcase-datasets/slurm/output/%j_%x.out
#SBATCH --error=experiments/034-showcase-datasets/slurm/output/%j_%x.out
#SBATCH --no-requeue

# Run the supported amino-acid + betaxanthin query against the served graph and write the
# showcase page's query fragment. The served store is queried only from a slurm job
# (precedent: experiments/025-solid-growth/scripts/gh_query_build_001.slurm). The three
# datasets serve about 14,000 experiments, fewer than the 21,498 records of the
# essentiality + SMF query (job 3029, MaxRSS 2.07 GB), so 32 G and 8 CPUs are a
# comfortable cap, not a measurement; MaxMemPerCPU=4100 makes 32 G imply the 8-CPU floor.
#
#   TORCHCELL_SRC=<checkout> sbatch experiments/034-showcase-datasets/scripts/gh_query_amino_acid_betaxanthin.slurm
#
# Submit from the root of the same checkout: the --output path is relative, so the
# log lands in that checkout's (git-ignored) experiments/034-showcase-datasets/slurm/.
# TORCHCELL_SRC selects the checkout (a worktree while the page is on a branch, the
# primary checkout once it lands); it defaults to the primary checkout. The build is a
# no-merge build (no converter, no deduplicator, GenotypeAggregator); a stale cache under
# $DATA_ROOT/data/torchcell/showcase_amino_acid_betaxanthin is reused as-is, so move it aside
# before a rebuild.

echo "Running on $(hostname), job $SLURM_JOB_ID"
TORCHCELL_SRC="${TORCHCELL_SRC:-/home/michaelvolk/Documents/projects/torchcell}"
cd "$TORCHCELL_SRC" || exit 1
echo "checkout: $TORCHCELL_SRC ($(git rev-parse --short HEAD))"

export PYTHONPATH="$TORCHCELL_SRC"
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="torchcell"

~/miniconda3/envs/torchcell/bin/python experiments/034-showcase-datasets/scripts/query_amino_acid_betaxanthin.py
