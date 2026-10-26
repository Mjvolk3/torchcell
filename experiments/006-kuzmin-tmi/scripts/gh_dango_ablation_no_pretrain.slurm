#!/bin/bash
#SBATCH -p main
#SBATCH --mem=24g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --job-name=dango-abl-no_pretrain
#SBATCH --time=2-00:00:00
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type=END
#SBATCH --output=/home/michaelvolk/Documents/projects/torchcell/experiments/006-kuzmin-tmi/slurm/output/%x_%j.out
#SBATCH --error=/home/michaelvolk/Documents/projects/torchcell/experiments/006-kuzmin-tmi/slurm/output/%x_%j.out

# experiments/006-kuzmin-tmi/scripts/gh_dango_ablation_no_pretrain.slurm
# DANGO ablation arm "no_pretrain" (conf/dango_ablation_no_pretrain.yaml), one GPU, 150 epochs,
# STRING v9.1, seed-42 split. See [[experiments.006-kuzmin-tmi.scripts.dango_ablation]].
#
# --time: 2 days is a placeholder, NOT a measurement. No GilaHyper card was idle on
# 2026-10-26 for a GPU smoke, so the epoch time is first measured by this job
# (perf/epoch_wall_s per epoch in W&B and on the "[epoch N]" lines of the log). If 150
# epochs do not fit, resume from <checkpoint dir>/last.ckpt with
# trainer.resume_ckpt_path=<path>.
# --mem: 24g is about 3x the 7.2 GB peak RSS of the CPU smoke (2026-10-26).
#
# Code runs from the fig/dango-si-completion worktree; the log goes to the primary
# checkout's (gitignored) slurm/output so it survives the worktree's removal. Do not
# land or remove the worktree while this job is PENDING: it reads the code at start.

cd /home/michaelvolk/Documents/projects/torchcell.worktrees/fig/dango-si-completion || exit 1

source ~/miniconda3/bin/activate
conda activate torchcell

export PYTHONPATH="/home/michaelvolk/Documents/projects/torchcell.worktrees/fig/dango-si-completion:$PYTHONPATH"
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS="ignore"

~/miniconda3/envs/torchcell/bin/python \
  experiments/006-kuzmin-tmi/scripts/dango_ablation.py \
  --config-name dango_ablation_no_pretrain
