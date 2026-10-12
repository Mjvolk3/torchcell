#!/bin/bash
# experiments/027-betaxanthin-metabolic/scripts/gh_cpu_build_essentiality.slurm
# [[experiments.027-betaxanthin-metabolic.wave9-essentiality]]
#
# Build the 027 gene-essentiality CellDataset from the served Neo4j graph.
#
# BLOCKED AS OF 2026-09-14 AND NOT SUBMITTED. The served store at bolt://localhost:7687 is
# reachable (the container is listening on 7687 and 7474) and EMPTY: 0 nodes, 0 Dataset
# nodes, 0 GeneEssentialitySgdDataset experiments, while
# $DATA_ROOT/database/kg_manifest.json still lists 36 served datasets including that one.
# The builder refuses an empty graph rather than writing an empty LMDB that reports success,
# so this cell would fail in its first seconds. Submit it once the store is repopulated.
#
# WHY A DEDICATED BUILD AT ALL: the two existing builds that contain the dataset converted
# `is_essential` into `fitness`, so neither can train a classifier on essentiality. See the
# query's header and [[experiments.027-betaxanthin-metabolic.wave9-essentiality]].
#
#   sbatch --nice=1000 experiments/027-betaxanthin-metabolic/scripts/gh_cpu_build_essentiality.slurm
#
#SBATCH -p main
#SBATCH --mem=16g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nice=1000
#SBATCH --job-name=027-build-essentiality
#SBATCH --time=04:00:00
#SBATCH --output=/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/kinetics-equilibrator-datasets/experiments/027-betaxanthin-metabolic/slurm/output/%x_%j.out
#SBATCH --error=/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/kinetics-equilibrator-datasets/experiments/027-betaxanthin-metabolic/slurm/output/%x_%j.out

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/kinetics-equilibrator-datasets}"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate torchcell
set -a; . "$PROJECT_ROOT/.env"; set +a
export EXPERIMENT_ROOT="$PROJECT_ROOT/experiments"
cd "$PROJECT_ROOT" || exit 1
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS=ignore
export NUM_WORKERS=0
echo "host $(hostname); DATA_ROOT=$DATA_ROOT; NEO4J_URI=${NEO4J_URI:-bolt://localhost:7687}"

python experiments/027-betaxanthin-metabolic/scripts/build_essentiality_dataset.py
