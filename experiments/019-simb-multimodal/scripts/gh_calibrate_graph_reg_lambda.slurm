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
cd "$PROJECT_ROOT/experiments" || exit 1

export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS=ignore
export WANDB_MODE=disabled

for LAM in 1e-8 1e-6 1e-4 1e-2; do
  echo "########## graph_reg_lambda=$LAM ##########"
  python "$PROJECT_ROOT/experiments/019-simb-multimodal/scripts/train_cgt_multitask.py" \
    --config-name cgt_expr_006 \
    model.graph_regularization.graph_reg_lambda=$LAM \
    model.hidden_channels=90 \
    model.num_transformer_layers=4 \
    multitask.dist=point \
    cell_dataset.node_embeddings=[random_100] \
    trainer.max_epochs=1 \
    trainer.early_stopping.enabled=false \
    data_module.num_workers=4 2>&1 | grep -E "^\[(train|val) epoch|graph_reg"
done
echo "########## DONE -- read ratio_to_data per lambda above ##########"
