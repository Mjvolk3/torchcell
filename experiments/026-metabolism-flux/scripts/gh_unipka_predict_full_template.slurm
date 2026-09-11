#!/bin/bash
# experiments/026-metabolism-flux/scripts/gh_unipka_predict_full_template.slurm
# Uni-pKa over the 300-compound validation set, five folds, one GPU, with the
# enumerator's FULL protonation template.
#
# Same computation as gh_unipka_predict.slurm, one setting apart: the microstate
# enumerator runs `smarts_pattern.tsv` rather than `simple_smarts_pattern.tsv`. The
# setting decides what the tool can possibly say. Measured on this sample, the simple
# template populates 2.01 adjacent macrostate pairs per compound and the full one 5.08,
# against a mean of 3.5 ChemAxon pKas, so under the simple template a 5-pKa or 6-pKa
# ladder is unreachable no matter how good the network is. The full template is also the
# enumerator's own default for `reconstruct`, the mode that built Uni-pKa's training sets.
#
# The cost is the CPU half: 19,603 microstates against the simple template's, each with
# 11 generated conformers, so the LMDB build is the long pole rather than the GPU passes.
# The charge window is widened to -6..+3 to match, though the sample never populates
# either extreme. The LMDB persists in $WORK_DIR, so a resubmit after a GPU failure skips
# the 45-minute conformer build and goes straight to inference.
#
#   cd <worktree-root> && sbatch experiments/026-metabolism-flux/scripts/gh_unipka_predict_full_template.slurm
#
#SBATCH -p main
#SBATCH --mem=64g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --job-name=026-unipka-full
#SBATCH --time=12:00:00
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/kinetics-equilibrator-datasets/experiments/026-metabolism-flux/slurm/output/%x_%j.out
#SBATCH --error=/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/kinetics-equilibrator-datasets/experiments/026-metabolism-flux/slurm/output/%x_%j.out

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/kinetics-equilibrator-datasets}"
mkdir -p "$PROJECT_ROOT/experiments/026-metabolism-flux/slurm/output"

ENV_FILE="$PROJECT_ROOT/.env"
[ -f "$ENV_FILE" ] || ENV_FILE="$HOME/Documents/projects/torchcell/.env"
set -a
. "$ENV_FILE"
set +a
export EXPERIMENT_ROOT="$PROJECT_ROOT/experiments"
echo "env $ENV_FILE; DATA_ROOT=$DATA_ROOT; EXPERIMENT_ROOT=$EXPERIMENT_ROOT"

cd "$PROJECT_ROOT" || exit 1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTHONUNBUFFERED=1
# The full template's ensembles reach 704 microstates for one compound, and the model pads
# an acid pool and a base pool into one tensor per row, so a batch of 16 such rows asked
# for 23.8 GB in one allocation and job 1672 died with a CUDA OOM after its LMDB was
# built. Batch 2 plus expandable segments is what fits on one 44 GiB card.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

UNIPKA_ROOT="$DATA_ROOT/unipka"
WORK_DIR="$DATA_ROOT/unipka/work_full"
mkdir -p "$WORK_DIR"

nvidia-smi
"$DATA_ROOT/envs/unipka/bin/python" \
    "$PROJECT_ROOT/experiments/026-metabolism-flux/scripts/external_pka_benchmark.py" \
    predict-unipka \
    --enumerator "$UNIPKA_ROOT/Uni-pKa/enumerator" \
    --model-dir "$UNIPKA_ROOT/models" \
    --work-dir "$WORK_DIR" \
    --tool-name unipka \
    --template smarts_pattern.tsv \
    --folds 0 1 2 3 4 \
    --charge-upper 3 \
    --charge-lower -6 \
    --batch-size 2 \
    --threads 8
