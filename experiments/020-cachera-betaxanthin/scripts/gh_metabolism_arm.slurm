#!/bin/bash
# experiments/020-cachera-betaxanthin/scripts/gh_metabolism_arm.slurm
# [[experiments.020-cachera-betaxanthin]]
#
# ONE metabolism arm on ONE GPU. Three of these run concurrently -- betaxanthin,
# beta-carotene, and the 19-AA metabolome -- one target each, so every arm is early-stopped
# and checkpointed on ITS OWN metric. That matters concretely: beta-carotene is a subjective
# ordinal and must be judged on Spearman, which a shared monitor cannot express.
#
#   METAB_ARM=betaxanthin   sbatch experiments/020-cachera-betaxanthin/scripts/gh_metabolism_arm.slurm
#   METAB_ARM=beta_carotene sbatch ...
#   METAB_ARM=mulleder19    sbatch ...
#
# W&B is ONLINE here, unlike the 019 pigment runs. Those were correctness checks whose
# artifact was a results JSON; these are keeper runs and the sweep view is the artifact.
#
#SBATCH -p main
#SBATCH --mem=96g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/scratch/projects/torchcell-scratch/experiments/020-cachera-betaxanthin/slurm/output/%x_%j.out
#SBATCH --error=/scratch/projects/torchcell-scratch/experiments/020-cachera-betaxanthin/slurm/output/%x_%j.out

set -euo pipefail

# Resolve the worktree root from the submit dir so the BRANCH's code is used and not the
# primary checkout's -- the editable install otherwise wins. Memory: worktree-build-needs-pythonpath.
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-/home/michaelvolk/Documents/projects/torchcell}"
mkdir -p /scratch/projects/torchcell-scratch/experiments/020-cachera-betaxanthin/slurm/output

source ~/miniconda3/etc/profile.d/conda.sh
conda activate torchcell
cd "$PROJECT_ROOT" || exit 1

export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS=ignore

ARM="${METAB_ARM:?set METAB_ARM to one of betaxanthin | beta_carotene | mulleder19}"

echo "=========================================================="
echo "host          : $(hostname)"
echo "job           : ${SLURM_JOB_ID:-none}  gpus=${CUDA_VISIBLE_DEVICES:-unset}"
echo "project root  : $PROJECT_ROOT"
echo "torchcell from: $(python -c 'import torchcell; print(torchcell.__file__)')"
echo "arm           : $ARM"
echo "wandb mode    : ${WANDB_MODE:-online}"
echo "=========================================================="
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
echo "=========================================================="

python experiments/020-cachera-betaxanthin/scripts/train_metabolism.py \
  --config-name="$ARM"

echo "=========================================================="
echo "DONE: $ARM"
echo "=========================================================="
