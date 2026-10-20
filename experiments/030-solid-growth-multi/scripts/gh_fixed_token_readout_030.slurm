#!/bin/bash
# experiments/030-solid-growth-multi/scripts/gh_fixed_token_readout_030.slurm
# [[experiments.030-solid-growth-multi.scripts.gh_fixed_token_readout_030]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/030-solid-growth-multi/scripts/gh_fixed_token_readout_030.slurm
#
# Read a trained 030 token checkpoint on the pinned triples under every token
# (fixed_token_readout_030.py): one GilaHyper GPU, one pass over the split.
#
#   CKPT=<...>/u28pgzde-last.ckpt TAG=r_s0_ep183 \
#     sbatch experiments/030-solid-growth-multi/scripts/gh_fixed_token_readout_030.slurm
#   CONFIG (default cgt_030_s3_r_tok_embfit_003) and SPLIT (default val) can be exported too.
#
# Memory is sized from the CPUs (GilaHyper's MaxMemPerCPU is 4.1 GB, see
# gh_smoke_dataset_token.slurm): one rank over the 13.5M-record build plus 10 workers.
#SBATCH -p main
#SBATCH --mem=48g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --job-name=030-fixed-token-readout
#SBATCH --time=06:00:00
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/030-per-entry-dataset-token/experiments/030-solid-growth-multi/slurm/output/%x_%j.out
#SBATCH --error=/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/030-per-entry-dataset-token/experiments/030-solid-growth-multi/slurm/output/%x_%j.out

PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
EXP="$PROJECT_ROOT/experiments/030-solid-growth-multi"
CONFIG="${CONFIG:-cgt_030_s3_r_tok_embfit_003}"
SPLIT="${SPLIT:-val}"
: "${CKPT:?export CKPT=<path to *-last.ckpt>}"
: "${TAG:?export TAG=<result suffix>}"
echo "job=$SLURM_JOB_ID name=$SLURM_JOB_NAME host=$(hostname) started=$(date)"
echo "project=$PROJECT_ROOT ($(git -C "$PROJECT_ROOT" rev-parse --short HEAD 2>/dev/null))"
echo "config=$CONFIG split=$SPLIT ckpt=$CKPT tag=$TAG"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate torchcell
set -a; source "$PROJECT_ROOT/.env"; set +a
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export EXPERIMENT_ROOT="$PROJECT_ROOT/experiments"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS="ignore"
echo "DATA_ROOT=$DATA_ROOT torchcell=$(python -c 'import torchcell; print(torchcell.__file__)')"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

cd "$PROJECT_ROOT/experiments" || exit 1
python "$EXP/scripts/fixed_token_readout_030.py" \
  --config "$CONFIG" --ckpt "$CKPT" --tag "$TAG" --split "$SPLIT" \
  --num-workers $((SLURM_CPUS_PER_TASK - 2))
STATUS=$?
echo "finished=$(date) exit=$STATUS"
exit $STATUS
