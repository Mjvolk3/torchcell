#!/bin/bash
# experiments/019-simb-multimodal/scripts/gh_metabolism_000.slurm
# [[plan.cgt-metabolism-flux-layer.2026.07.26]]
#
# CGT-Metabolism -- ONE model over betaxanthin + the 19-AA metabolome, full graph block.
#
# This is NOT the four-condition Delta driver (`run_pigment_conditions.py`). That driver
# trains four separate arms to difference their metrics; this trains a SINGLE joint model,
# which is what the betaxanthin objective actually calls for. It therefore invokes
# `train_cgt_multitask.py` directly with `--config-name`, bypassing the driver's
# PIGMENT_* env contract and its resume-from-partial checkpoint entirely.
#
# Config `gh_metabolism_000.yaml` carries the reasoning; the three changes that matter:
#   * NINE graphs (010's block), not two -- 019 ran two and that was the mistake.
#   * NO learnable embedding table; `prot_T5_all` content features instead (top of the
#     kNN embedding probe at 0.0704 morph / 0.1174 expr, with zero learned parameters).
#   * graph_reg_lambda RE-TUNED to 6.5e-5. The old 1e-3 was tuned while lambda was being
#     squared; with that fixed, 1e-3 puts the prior at 3.85x the data loss. 6.5e-5 puts it
#     at ~25%, verified on a real batch (val/graph_reg/ratio_to_data = 0.251).
#
# beta-carotene is deliberately OUT: subjective ordinal with ~0.54 replicate rank
# agreement, and Mulleder measures none of its precursors, so there is no transfer path.
#
#SBATCH -p main
#SBATCH --mem=96g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --job-name=019-metab
#SBATCH --time=12:00:00
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/scratch/projects/torchcell-scratch/experiments/019-simb-multimodal/slurm/output/%x_%j.out
#SBATCH --error=/scratch/projects/torchcell-scratch/experiments/019-simb-multimodal/slurm/output/%x_%j.out

set -euo pipefail

# Resolve the worktree root from the submit dir so the BRANCH's code is used, not the
# primary checkout's -- the editable install otherwise wins. See memory
# `worktree-build-needs-pythonpath`.
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-/home/michaelvolk/Documents/projects/torchcell}"
mkdir -p /scratch/projects/torchcell-scratch/experiments/019-simb-multimodal/slurm/output

source ~/miniconda3/etc/profile.d/conda.sh
conda activate torchcell
cd "$PROJECT_ROOT" || exit 1

export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS=ignore
export WANDB_MODE=offline

CONFIG="${METAB_CONFIG:-gh_metabolism_000}"

echo "=========================================================="
echo "host          : $(hostname)"
echo "job           : ${SLURM_JOB_ID:-none}  gpus=${CUDA_VISIBLE_DEVICES:-unset}"
echo "project root  : $PROJECT_ROOT"
echo "torchcell from: $(python -c 'import torchcell; print(torchcell.__file__)')"
echo "config        : $CONFIG"
echo "=========================================================="
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
echo "=========================================================="

python experiments/019-simb-multimodal/scripts/train_cgt_multitask.py \
  --config-name="$CONFIG"

echo "=========================================================="
echo "DONE."
echo "=========================================================="
