#!/bin/bash
# experiments/028-gene-essentiality/scripts/gh_sampled_flux_store.slurm
# [[experiments.028-gene-essentiality.scripts.solid_growth_store]]
#
# The module's store over the solid-growth build: one interior-point solve per distinct
# (GEM gene subset, medium) key, in order of records served. CPU only; 0.15 s a program
# on one thread, so a million keys is about two hours at twenty workers.
#
#   TC_HIGHSPY_PATH=$DATA_ROOT/tools/highspy-1.15.1 ESS_MEM=64g ESS_CPUS=20 \
#     STORE_ARGS="--medium SC --solver ipm" \
#     bash experiments/028-gene-essentiality/scripts/freeze_and_submit.sh \
#       experiments/028-gene-essentiality/scripts/gh_solid_growth_store.slurm
#
#SBATCH -p main
#SBATCH --mem=48g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=028-sample
#SBATCH --time=24:00:00
#SBATCH --output=/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/kinetics-equilibrator-datasets/experiments/028-gene-essentiality/slurm/output/%x_%j.out
#SBATCH --error=/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/kinetics-equilibrator-datasets/experiments/028-gene-essentiality/slurm/output/%x_%j.out

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/kinetics-equilibrator-datasets}"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate torchcell
set -a; . "$PROJECT_ROOT/.env"; set +a
export EXPERIMENT_ROOT="$PROJECT_ROOT/experiments"
cd "$PROJECT_ROOT" || exit 1
# No standalone highspy here: the sampler is cobra's OptGPSampler on the packaged solver.
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS=ignore
echo "host $(hostname); processes $SLURM_CPUS_PER_TASK; args ${STORE_ARGS:-}"
python experiments/028-gene-essentiality/scripts/ess_sampled_flux_store.py \
    --processes "$SLURM_CPUS_PER_TASK" ${STORE_ARGS:-}
