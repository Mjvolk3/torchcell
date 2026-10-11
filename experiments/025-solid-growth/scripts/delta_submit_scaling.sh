#!/bin/bash
# experiments/025-solid-growth/scripts/delta_submit_scaling.sh
# [[experiments.025-solid-growth.scripts.delta_submit_scaling]]
#
# The first scaling ablation on the trigenic build, on Delta, as 1-GPU jobs under the
# 48 h limit (notes-tex/modeling/scaling-laws, Section 5). Every run is the closure arm on
# the query-pair-disjoint split, cgt_s3_q_kl_fit_036 (strict pool S3Q, learnable table,
# AdamW 2.5e-4 constant, 50 epochs), with the size or the training subset changed:
#
#   N ladder   width x depth scaled together at 9 heads (the width must divide by 9):
#              45x2, 63x3, 90x4, 126x6, 180x8. Non-embedding parameters about 0.05M,
#              0.14M, 0.39M, 1.1M, 3.1M (12 h^2 L for the transformer stack, measured
#              per run by ComputeAccounting). 180x8 is the current model and the
#              held-out largest point of the fit. Three seeds at the two smallest sizes.
#   LR check   45x2 at 1e-4 and 1e-3 beside the 2.5e-4 default, so the smallest size's
#              optimum is known before the curve is read.
#   D ladder   the current size on nested S3Q training subsets drawn by query pair,
#              1/64, 1/16, 1/4 (subset_s3q_fractions.py); the 180x8 ladder job is the 1.
#
# One GPU per job so the jobs backfill ahead of 4-GPU requests. Each job stages the 554 GB
# LMDB onto node-local NVMe (delta_cgt.slurm); two staged jobs fit on one A40 node's local
# disk, a third fails fast at the free-space check and can be resubmitted. --mem=100g per
# job, four to a node at most. Clock: 48 h. Measured Delta pace for this trainer is 23 min
# per 377k-record epoch per A40 (4-GPU DDP, NVMe-staged), so S3Q's 1.05M training records
# run at about 64 min per epoch on one A40 and 48 h holds about 44 epochs; the 10-29 and
# most of the 30-49 scoring windows complete at every size. Smaller sizes are at least as
# fast. GPU hours: 14 jobs x at most 48 = 672 against bfjt-delta-gpu.
#
#   RUN ON A DELTA LOGIN NODE, FROM THE WORKTREE ROOT (the job-owned clone, never the
#   shared /projects checkout), after delta_preflight_025.sh.
#
#   DRY=1 bash experiments/025-solid-growth/scripts/delta_submit_scaling.sh    # print only
#         bash experiments/025-solid-growth/scripts/delta_submit_scaling.sh    # submit
#   LANES=2 staggers the stage-ins: consecutive jobs in a lane start 30 s apart.
set -euo pipefail
ACCOUNT="${ACCOUNT:-bfjt-delta-gpu}"
LAUNCHER="experiments/025-solid-growth/scripts/delta_cgt.slurm"
CONFIG="cgt_s3_q_kl_fit_036"
HOURS="${HOURS:-48}"
MEM="${MEM:-100g}"
CPUS="${CPUS:-16}"
LANES="${LANES:-2}"
DRY="${DRY:-0}"
[[ -f "$LAUNCHER" ]] || { echo "run from the worktree root: $LAUNCHER not found" >&2; exit 2; }
[[ -f "experiments/025-solid-growth/conf/$CONFIG.yaml" ]] || { echo "missing $CONFIG" >&2; exit 2; }
for f in subset_S3Q_indices subset_S3Q_frac4_indices subset_S3Q_frac16_indices subset_S3Q_frac64_indices; do
  [[ -f "experiments/025-solid-growth/results/$f.json.gz" ]] || { echo "missing results/$f.json.gz" >&2; exit 2; }
done

declare -a prev
for ((i = 0; i < LANES; i++)); do prev[$i]=""; done
lane=0
submit() {  # submit <job-name> [overrides...]
  local name="$1"; shift
  local cmd=(sbatch --parsable --account="$ACCOUNT" --time="${HOURS}:00:00" -J "$name"
             --gpus-per-node=1 --cpus-per-task="$CPUS" --mem="$MEM" --export=ALL,GPUS=1)
  [[ -n "${prev[$lane]}" ]] && cmd+=(--dependency="after:${prev[$lane]}+30")
  cmd+=("$LAUNCHER" "$CONFIG" trainer.devices=1 "$@")
  echo "${cmd[*]}"
  if [[ "$DRY" == "1" ]]; then
    prev[$lane]="DRY-$name"
  else
    prev[$lane]=$("${cmd[@]}")
    echo "  -> ${prev[$lane]}"
  fi
  lane=$(( (lane + 1) % LANES ))
}

# N ladder, smallest first so the cheap points land early; three seeds at the two smallest.
for s in 1 2 3; do
  submit "025-sclw1-N45L2-s$s"  model.hidden_channels=45  model.num_transformer_layers=2 +seed="$s"
  submit "025-sclw1-N63L3-s$s"  model.hidden_channels=63  model.num_transformer_layers=3 +seed="$s"
done
submit "025-sclw1-N90L4-s1"   model.hidden_channels=90  model.num_transformer_layers=4 +seed=1
submit "025-sclw1-N126L6-s1"  model.hidden_channels=126 model.num_transformer_layers=6 +seed=1
submit "025-sclw1-N180L8-s1"  model.hidden_channels=180 model.num_transformer_layers=8 +seed=1
# LR check at the smallest size.
submit "025-sclw1-N45L2-lr1e-4-s1" model.hidden_channels=45 model.num_transformer_layers=2 regression_task.optimizer.lr=1e-4 +seed=1
submit "025-sclw1-N45L2-lr1e-3-s1" model.hidden_channels=45 model.num_transformer_layers=2 regression_task.optimizer.lr=1e-3 +seed=1
# D ladder at the current size; the full point is 025-sclw1-N180L8-s1.
submit "025-sclw1-D64-s1" subset.indices=subset_S3Q_frac64_indices.json.gz +seed=1
submit "025-sclw1-D16-s1" subset.indices=subset_S3Q_frac16_indices.json.gz +seed=1
submit "025-sclw1-D4-s1"  subset.indices=subset_S3Q_frac4_indices.json.gz  +seed=1
echo "lanes end at: ${prev[*]}"
