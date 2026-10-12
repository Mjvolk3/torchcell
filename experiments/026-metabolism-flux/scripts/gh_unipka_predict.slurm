#!/bin/bash
# experiments/026-metabolism-flux/scripts/gh_unipka_predict.slurm
# Uni-pKa over the 300-compound validation set, five folds, one GPU, with the
# enumerator's SIMPLE protonation template.
#
# Uni-pKa is the only tool in this benchmark that produces a MACROSCOPIC constant itself,
# from the free energies of the microstates in two adjacent macrostates, so it is the one
# tool whose output needs no ladder aggregation from this side. The work is therefore
# split in two: the repository's own microstate enumerator builds the protonation ensemble
# on CPU, and the Uni-Mol network prices every macrostate pair on GPU.
#
# `simple_smarts_pattern.tsv` is the enumerator's default for `ensemble` and discards the
# rarer solution motifs. It admits 2.01 protonation steps per compound on this sample
# against ChemAxon's 3.5 pKas, so it cannot express the longer ladders by construction;
# the companion script runs the full template, which admits 5.08.
#
# Weights: `bindresearch/pkasso_unipka` folds 0 to 4 on HuggingFace, sha256-pinned by the
# `unipkainfer` package's own registry, staged at $DATA_ROOT/unipka/models. Enumerator:
# github.com/dptech-corp/Uni-pKa at 7cfbf6532d7e14f427abaed5859bd1001b4b6377, staged at
# $DATA_ROOT/unipka/Uni-pKa.
#
#   cd <worktree-root> && sbatch experiments/026-metabolism-flux/scripts/gh_unipka_predict.slurm
#
#SBATCH -p main
#SBATCH --mem=48g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --job-name=026-unipka-simple
#SBATCH --time=08:00:00
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/kinetics-equilibrator-datasets/experiments/026-metabolism-flux/slurm/output/%x_%j.out
#SBATCH --error=/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/kinetics-equilibrator-datasets/experiments/026-metabolism-flux/slurm/output/%x_%j.out

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/kinetics-equilibrator-datasets}"
mkdir -p "$PROJECT_ROOT/experiments/026-metabolism-flux/slurm/output"

# `load_dotenv()` searches upward from CWD, and a worktree's `.env` is gitignored and
# machine-specific, so resolve it explicitly rather than relying on the search.
ENV_FILE="$PROJECT_ROOT/.env"
[ -f "$ENV_FILE" ] || ENV_FILE="$HOME/Documents/projects/torchcell/.env"
set -a
. "$ENV_FILE"
set +a
# Pin the experiment root to THIS worktree, so the prediction file lands beside the
# script that made it rather than in whichever checkout the shared `.env` names.
export EXPERIMENT_ROOT="$PROJECT_ROOT/experiments"
echo "env $ENV_FILE; DATA_ROOT=$DATA_ROOT; EXPERIMENT_ROOT=$EXPERIMENT_ROOT"

cd "$PROJECT_ROOT" || exit 1
# Eight threads, matching --cpus-per-task: the conformer generation that builds the LMDB
# is the CPU half of this job and will otherwise spawn a thread per core on the node.
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTHONUNBUFFERED=1

UNIPKA_ROOT="$DATA_ROOT/unipka"
WORK_DIR="$DATA_ROOT/unipka/work_simple"
mkdir -p "$WORK_DIR"

nvidia-smi
"$DATA_ROOT/envs/unipka/bin/python" \
    "$PROJECT_ROOT/experiments/026-metabolism-flux/scripts/external_pka_benchmark.py" \
    predict-unipka \
    --enumerator "$UNIPKA_ROOT/Uni-pKa/enumerator" \
    --model-dir "$UNIPKA_ROOT/models" \
    --work-dir "$WORK_DIR" \
    --tool-name unipka-simple-template \
    --template simple_smarts_pattern.tsv \
    --folds 0 1 2 3 4 \
    --charge-upper 2 \
    --charge-lower -4 \
    --batch-size 16 \
    --threads 8
