#!/bin/bash
# experiments/019-simb-multimodal/scripts/gh_calibrate_graph_reg_lambda.slurm
# Measure the graph-regularization strength implied by each candidate lambda, BEFORE
# committing the 3-day IGB window to a grid.
#
# WHY THIS EXISTS. lambda is not portable -- it is only meaningful relative to a specific
# normalization and a specific number of regularized layers. Both changed:
#   * the normalization is now main's per-graph EDGE-COUNT form (lambda applied once),
#     not the per-graph mean-degree form this branch had calibrated against;
#   * graph_reg_layer went from all 8 layers to a SINGLE layer (010's setting), which
#     shrinks the summed KL by ~L.
# Nothing errors when lambda is wrong by orders of magnitude -- the first _006 launch ran
# with the graph term at 99.99% of the objective and looked perfectly healthy. So measure.
#
# READS `val/graph_reg/ratio_to_data` = graph_term / data_term. 1.0 is parity; the grid
# should bracket roughly 0.01 (barely on) to ~10 (dominant), centred near parity.
#
# One epoch per lambda, single GPU, minutes.
#
#   cd <worktree-root> && sbatch experiments/019-simb-multimodal/scripts/gh_calibrate_graph_reg_lambda.slurm
#
#SBATCH -p main
#SBATCH --mem=90g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --job-name=019-lambda-cal
#SBATCH --time=04:00:00
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/scratch/projects/torchcell-scratch/experiments/019-simb-multimodal/slurm/output/%x_%j.out
#SBATCH --error=/scratch/projects/torchcell-scratch/experiments/019-simb-multimodal/slurm/output/%x_%j.out

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-/home/michaelvolk/Documents/projects/torchcell}"
mkdir -p /scratch/projects/torchcell-scratch/experiments/019-simb-multimodal/slurm/output

source ~/miniconda3/etc/profile.d/conda.sh
conda activate torchcell

# train_cgt_multitask.py calls bare `load_dotenv()`, which searches UPWARD FROM CWD. A
# git worktree has no `.env` unless setup-worktree.sh created one (it is gitignored and
# machine-specific), so a run launched from a worktree dies at import on
# "DATA_ROOT environment variable is not set". Resolve it explicitly: worktree first,
# primary checkout as fallback.
ENV_FILE="$PROJECT_ROOT/.env"
[ -f "$ENV_FILE" ] || ENV_FILE="$HOME/Documents/projects/torchcell/.env"
set -a
. "$ENV_FILE"
set +a
echo "sourced env from $ENV_FILE (DATA_ROOT=$DATA_ROOT)"

cd "$PROJECT_ROOT/experiments" || exit 1

export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS=ignore
export WANDB_MODE=disabled
export HYDRA_FULL_ERROR=1

python "$PROJECT_ROOT/experiments/019-simb-multimodal/scripts/calibrate_graph_reg_lambda.py"
