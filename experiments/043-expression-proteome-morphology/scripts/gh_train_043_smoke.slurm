#!/bin/bash
# experiments/043-expression-proteome-morphology/scripts/gh_train_043_smoke.slurm
# [[experiments.043-expression-proteome-morphology.scripts.gh_train_043_smoke]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/043-expression-proteome-morphology/scripts/gh_train_043_smoke.slurm
#
# GPU smoke cell of the 043 trainer: joint_9, HL-Gauss, equal weighting, seed 0, batch 8, one
# card. Epochs = min(50, what fits the 7.6 h budget by the MEASURED first epoch (training and
# evaluation), less a 20 min margin); the projection from the first 30 steps is printed at
# startup and logged (setup/probe_seconds_per_step, setup/projected_train_minutes_per_epoch),
# and the fixed epoch count and registered window after epoch 1 ([budget] line).
#
# Needs the prepared store (gh_train_043_prepare.slurm) and every vector the arm's records
# use in alleles/<encoder>.h5: chain it behind the embedding chain (6106 -> 6115 -> 6121 ->
# 6122). A key with no vector raises KeyError at startup; nothing is substituted.
#
#   sbatch --dependency=afterany:6122 experiments/043-expression-proteome-morphology/scripts/gh_train_043_smoke.slurm
#
# Twins (prepared, not submitted): pass ARM / LOSS / WEIGHTING through the environment, e.g.
#   sbatch --export=ALL,ARM=alone_kemmeren --dependency=afterany:<smoke> .../gh_train_043_smoke.slurm
#   sbatch --export=ALL,LOSS=pinball --dependency=afterany:<previous> .../gh_train_043_smoke.slurm
#   sbatch --export=ALL,LOSS=mse --dependency=afterany:<previous> .../gh_train_043_smoke.slurm
# Dataset-balanced sampler (data.sampler=balanced; W&B group joint_9_balanced_hlgauss_equal):
#   sbatch --export=ALL,SAMPLER=balanced --job-name=043-train-smoke-bal --dependency=afterany:6122 .../gh_train_043_smoke.slurm
#
# Overnight test queue (2026-10-22; two chains, afterany, budget matched to --time):
#   chain 1 after 6136:
#   sbatch --export=ALL,LOSS=pinball,TIME_BUDGET_HOURS=4.3,DEADLINE=1792758540 --job-name=043-train-pinball --time=04:40:00 --dependency=afterany:6136 .../gh_train_043_smoke.slurm
#   sbatch --export=ALL,ARM=alone_kemmeren,MAX_STEPS=317300,EVAL_EVERY=42,TIME_BUDGET_HOURS=4.5,DEADLINE=1792758540 --job-name=043-train-alone-kem --time=04:50:00 --dependency=afterany:<pinball> .../gh_train_043_smoke.slurm
#   chain 2 after 6137:
#   sbatch --export=ALL,WEIGHTING=records,TIME_BUDGET_HOURS=4.3,DEADLINE=1792758540 --job-name=043-train-records --time=04:40:00 --dependency=afterany:6137 .../gh_train_043_smoke.slurm
#   (2b, LOSS=mse, dropped: 6137's balanced sampler runs 0.074 s per step, so chain 2
#   reaches 2a only near 04:25 and the 10 h window ends 07:29; DEADLINE is that end)
# MAX_STEPS=317300 is 50 epochs x 6,346 steps of joint_9 (job 6136, batch 8); Kemmeren
# alone has 1,202 training records = 151 steps per epoch, so 2,102 epochs, evaluated every
# 42 (about 50 evaluations).
#
# --mem: host-side peak is the prepared store (measured at prepare time) plus one encoder's
# full h5 embedding array while the compact table is gathered (FUDT upstream 1.21 M x 768
# fp16 = 1.9 GB) plus the compact tables before they move to the card (about 5.8 GB if every
# vocabulary row is used); 48g covers it with the forked loader workers. Measure the real
# peak with `sacct -j <id> --format=JobID,MaxRSS -P` and tighten.
#SBATCH -p main
#SBATCH --gres=gpu:1
#SBATCH --mem=48g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --job-name=043-train-smoke
#SBATCH --time=08:00:00
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/scratch/projects/torchcell-scratch/experiments/043-expression-proteome-morphology/slurm/output/%x_%j.out
#SBATCH --error=/scratch/projects/torchcell-scratch/experiments/043-expression-proteome-morphology/slurm/output/%x_%j.out
set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate torchcell
cd "${SLURM_SUBMIT_DIR:?submit from the repository root}"
export PYTHONPATH="$PWD"
export DATA_ROOT=/scratch/projects/torchcell-scratch
export EXPERIMENT_ROOT="$PWD/experiments"
ARM="${ARM:-joint_9}"
LOSS="${LOSS:-hlgauss}"
WEIGHTING="${WEIGHTING:-equal}"
SEED="${SEED:-0}"
SAMPLER="${SAMPLER:-records}"
# the trainer's own budget must match this job's --time (7.6 h for the 8 h default)
TIME_BUDGET_HOURS="${TIME_BUDGET_HOURS:-7.6}"
# DEADLINE (epoch seconds): the run must end by then; the trainer's budget becomes the
# smaller of TIME_BUDGET_HOURS and the hours left, so the epoch count shrinks to fit
if [ -n "${DEADLINE:-}" ]; then
  LEFT=$(python -c "import time; print(round(max(0.1, (${DEADLINE} - time.time()) / 3600), 2))")
  TIME_BUDGET_HOURS=$(python -c "print(min(${TIME_BUDGET_HOURS}, ${LEFT}))")
fi
EVAL_EVERY="${EVAL_EVERY:-1}"
MAX_EPOCHS="${MAX_EPOCHS:-50}"
TIME_MARGIN_MINUTES="${TIME_MARGIN_MINUTES:-20}"
# marks per record: null = every mark, the default since 2026-10-23 (owner: "bump the
# limit"; data maximum 4,416, N'Guessan). A number is a smoke-speed override; the 2026-10-22
# runs and S1 to S4 (6172 to 6175) used 1024.
MARK_CAP="${MARK_CAP:-null}"
TAGS="${TAGS:-smoke}"
# step-matched alone runs: total optimizer steps of the joint run they control for
EXTRA=()
if [ -n "${MAX_STEPS:-}" ]; then EXTRA+=("budget.max_steps=${MAX_STEPS}"); fi
# stage 2: the stage-1 checkpoint (model weights only), its lr and warmup
if [ -n "${INIT_FROM:-}" ]; then EXTRA+=("init_from=${INIT_FROM}"); fi
if [ -n "${LR:-}" ]; then EXTRA+=("optim.lr=${LR}"); fi
if [ -n "${WARMUP_STEPS:-}" ]; then EXTRA+=("optim.warmup_steps=${WARMUP_STEPS}"); fi
echo "host $(hostname)  commit $(git rev-parse --short HEAD)  $(date)"
echo "arm $ARM  loss $LOSS  weighting $WEIGHTING  sampler $SAMPLER  seed $SEED  mark_cap $MARK_CAP  max_epochs $MAX_EPOCHS  max_steps ${MAX_STEPS:-none}  init_from ${INIT_FROM:-none}  lr ${LR:-default}  eval_every $EVAL_EVERY  budget ${TIME_BUDGET_HOURS} h"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv
python experiments/043-expression-proteome-morphology/scripts/train_043.py \
  arm="$ARM" loss="$LOSS" weighting="$WEIGHTING" seed="$SEED" \
  data.batch_size=8 data.num_workers=4 data.sampler="$SAMPLER" data.mark_cap="$MARK_CAP" \
  budget.max_epochs="$MAX_EPOCHS" budget.time_budget_hours="$TIME_BUDGET_HOURS" budget.time_margin_minutes="$TIME_MARGIN_MINUTES" \
  budget.eval_every_epochs="$EVAL_EVERY" "${EXTRA[@]}" \
  "wandb.tags=[${TAGS}]"
echo "done $(date)"
