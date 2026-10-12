#!/bin/bash
# experiments/027-betaxanthin-metabolic/scripts/gh_bisect_a2a17438.slurm
# [[experiments.027-betaxanthin-metabolic]]
#
# BISECT (2026-09-09). Both positive controls on HEAD collapse (jobs 1616, 1618), and the
# 020 Delta grid trained on commit 51fe7ec4 (2026-08-01) with pred_sd_ratio 0.8. This job
# runs the 020 config (delta_grid_betaxanthin.yaml, hard mask, L=6) on the 019 trainer from
# a detached worktree of commit a2a17438, 12 epochs, one GPU. Three commits are run side by
# side: 51fe7ec4 (the August baseline), a2a17438 (after the data streaming and dedup
# changes of 08-31 to 09-01), be0b8010 (after the 09-02 change to the transformer).
#
#   cd <worktree-root> && sbatch experiments/027-betaxanthin-metabolic/scripts/gh_bisect_a2a17438.slurm
#
#SBATCH -p main
#SBATCH --mem=12g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --job-name=027-bisect-a2a17438
#SBATCH --time=01:30:00
#SBATCH --output=/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/kinetics-equilibrator-datasets/experiments/027-betaxanthin-metabolic/slurm/output/%x_%j.out
#SBATCH --error=/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/kinetics-equilibrator-datasets/experiments/027-betaxanthin-metabolic/slurm/output/%x_%j.out

PROJECT_ROOT=/scratch/projects/torchcell-scratch/bisect/wt-a2a17438
source ~/miniconda3/etc/profile.d/conda.sh
conda activate torchcell
set -a; . "$PROJECT_ROOT/.env"; set +a
export EXPERIMENT_ROOT="$PROJECT_ROOT/experiments"
cd "$PROJECT_ROOT/experiments" || exit 1
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS=ignore
export WANDB__SERVICE_WAIT=600
export WANDB_DIR=/scratch/projects/torchcell-scratch/experiments/027-betaxanthin-metabolic/wandb
mkdir -p "$WANDB_DIR"
echo "host $(hostname) gpu $CUDA_VISIBLE_DEVICES; tree $PROJECT_ROOT @ $(git -C $PROJECT_ROOT rev-parse --short HEAD)"
python -c "import torchcell, os; print('torchcell from', os.path.dirname(torchcell.__file__))"

python "$PROJECT_ROOT/experiments/019-simb-multimodal/scripts/train_cgt_multitask.py" \
  --config-path "$PROJECT_ROOT/experiments/020-cachera-betaxanthin/conf" --config-name delta_grid_betaxanthin \
  trainer.max_epochs=12 trainer.devices=1 data_module.num_workers=3 \
  wandb.project=torchcell_027_bxfx wandb.tags='[027,bisect,a2a17438]'
