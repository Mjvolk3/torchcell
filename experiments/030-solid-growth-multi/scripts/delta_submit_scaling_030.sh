#!/bin/bash
# experiments/030-solid-growth-multi/scripts/delta_submit_scaling_030.sh
# [[experiments.030-solid-growth-multi.scripts.delta_submit_scaling_030]]
#
# The scaling grid of the 030 per-entry CGT on Delta: one GPU per job, the S3Q pool store
# staged per job (delta_cgt_030.slurm), account bgcg-delta-gpu. Every run is the Q-split
# closure arm cgt_030_s3_q_tok_embfit_004 (composite + token, strict pool S3Q) scored on
# the pinned 37,705 validation triples by per-screen MSE in label units (the readout,
# scaling_readout_030.py); what varies is the size and the training subset:
#
#   N ladder   width x depth at 9 heads: 45x2, 63x3, 90x4, 126x6, 180x8 (the current size).
#   D ladder   nested subsets by query pair (subset_s3q_fractions_030.py): 1/64, 1/16,
#              1/4 and the full S3Q; D in gene_interaction entry rows 33k / 132k / 509k /
#              2.34M (records 21k / 69k / 244k / 1.05M). Every subset uses the FULL pool's
#              normalization constants (transforms.fit_stats_strict=false).
#   grid       all 5 sizes x all 4 subsets, seed 1; seeds 2 and 3 at 1/64 and 1/16 for the
#              two end sizes; a learning-rate check (1e-4, 1e-3) at 45x2 on 1/4.
#
# EPOCH BUDGETS. Measured pace: one A40 processes about 68 records/s at the current size
# (4-GPU 025 runs: 377k records in 23 min), so the full S3Q epoch is about 4.3 h and a
# 48 h job holds about 11 of them. Every point fits ONE job, because Slurm accrues no
# queue priority while a job waits on a dependency and the A40 queue runs days, so a
# chain of links pays a fresh queue wait per link:
#   1/64: 240 epochs   1/16: 120 epochs   1/4: 40 epochs   full: 10 epochs
# (full and 1/4 are what 48 h holds at 180x8; smaller sizes finish sooner). The small
# subsets converge inside their budget; the large ones are read at a matched records-seen
# budget, which the readout interpolates from the per-epoch validation loss. The ONE
# chained point is 180x8 on the full set, the headline: 3 links of 48 h resuming the full
# training state (RESUME_FROM_JOB, --dependency=afterany) for 30 epochs. CHAIN_HEADLINE=0
# makes it a single 10-epoch job like the rest.
#
# BATCH. 256 records per step on one GPU for every run (the IGB 180x8 full-data run uses
# 2 x 512, so it is NOT a point of this grid; the grid runs its own 180x8 full-data chain).
#
# ORDER. Submit ONLY=calib first: two 3 h jobs, 180x8 and 45x2 on the full pool, 400
# training batches each, which log compute/records_seen, compute/gpu_hours_cumulative
# and the measured FLOPs per step; the records/s at 180x8 is the pace every budget
# above assumes, and the 45x2 pace says whether the loader or the GPU binds (equal
# paces mean the loader, and then no size finishes sooner than 180x8). Then ONLY=grid.
# The config defines transforms.fit_stats_strict so the plain override composes; the
# configs and the code are read from the checkout when a job STARTS, so a pull on
# Delta reaches every job still pending.
#
#   RUN ON A DELTA LOGIN NODE, FROM THE WORKTREE ROOT, after `git pull`:
#   mkdir -p /scratch/bbub/mjvolk3/torchcell/experiments/030-solid-growth-multi/slurm/output
#   DRY=1 ONLY=calib bash experiments/030-solid-growth-multi/scripts/delta_submit_scaling_030.sh
#         ONLY=calib bash experiments/030-solid-growth-multi/scripts/delta_submit_scaling_030.sh
#         ONLY=grid  bash experiments/030-solid-growth-multi/scripts/delta_submit_scaling_030.sh
set -euo pipefail
ACCOUNT="${ACCOUNT:-bgcg-delta-gpu}"
LAUNCHER="experiments/030-solid-growth-multi/scripts/delta_cgt_030.slurm"
CONFIG="cgt_030_s3_q_tok_embfit_004"
SWEEP="${SWEEP:-030_sclw1}"
HOURS="${HOURS:-48}"
MEM="${MEM:-80g}"
CPUS="${CPUS:-16}"
BATCH="${BATCH:-256}"
ONLY="${ONLY:-grid}"
DRY="${DRY:-0}"
[[ -f "$LAUNCHER" ]] || { echo "run from the worktree root: $LAUNCHER not found" >&2; exit 2; }
[[ -f "experiments/030-solid-growth-multi/conf/$CONFIG.yaml" ]] || { echo "missing $CONFIG" >&2; exit 2; }
for f in subset_S3Q_indices_030 subset_S3Q_frac4_indices_030 subset_S3Q_frac16_indices_030 subset_S3Q_frac64_indices_030; do
  [[ -f "experiments/030-solid-growth-multi/results/$f.json.gz" ]] || { echo "missing results/$f.json.gz" >&2; exit 2; }
done

COMMON=(trainer.devices=1 "data_module.batch_size=$BATCH" transforms.fit_stats_strict=false "+scaling.sweep=$SWEEP")

# submit <name> <hours> <dependency-job-or-empty> [overrides...]; prints the job id
submit() {
  local name="$1" hours="$2" dep="$3"; shift 3
  local cmd=(sbatch --parsable --account="$ACCOUNT" --time="${hours}:00:00" -J "$name"
             --gpus-per-node=1 --cpus-per-task="$CPUS" --mem="$MEM")
  if [[ -n "$dep" ]]; then
    cmd+=(--dependency="afterany:$dep" --export="ALL,GPUS=1,RESUME_FROM_JOB=$dep")
  else
    cmd+=(--export=ALL,GPUS=1)
  fi
  cmd+=("$LAUNCHER" "$CONFIG" "${COMMON[@]}" "$@")
  echo "${cmd[*]}" >&2
  if [[ "$DRY" == "1" ]]; then echo "DRY-$name"; else "${cmd[@]}"; fi
}

# chain <name> <links> <hours> [overrides...]: link k resumes link k-1
chain() {
  local name="$1" links="$2" hours="$3"; shift 3
  local prev="" id
  for ((k = 1; k <= links; k++)); do
    id=$(submit "$name-c$k" "$hours" "$prev" "$@")
    echo "  -> $id" >&2
    prev="$id"
  done
}

size() {  # size <width> <depth>
  echo "model.hidden_channels=$1 model.num_transformer_layers=$2"
}
sub() {   # sub <denominator|full>
  if [[ "$1" == "full" ]]; then echo "subset.indices=subset_S3Q_indices_030.json.gz"
  else echo "subset.indices=subset_S3Q_frac$1_indices_030.json.gz"; fi
}
declare -A EPOCHS=([64]=240 [16]=120 [4]=40 [full]=10)
declare -A LINKS=([64]=1 [16]=1 [4]=1 [full]=1)
CHAIN_HEADLINE="${CHAIN_HEADLINE:-0}"

if [[ "$ONLY" == "calib" ]]; then
  for wd in "180 8" "45 2"; do
    set -- $wd
    submit "030-sclw1-calib-N$1L$2" 3 "" $(size "$1" "$2") $(sub full) +seed=1 trainer.max_epochs=1 trainer.limit_train_batches=400 >/dev/null
  done
  echo "calibration submitted (180x8 and 45x2); read compute/records_seen, compute/gpu_hours_cumulative and compute/flops_per_step from their W&B runs"
  exit 0
fi

for d in 64 16 4 full; do
  for wd in "45 2" "63 3" "90 4" "126 6" "180 8"; do
    set -- $wd
    links="${LINKS[$d]}"; epochs="${EPOCHS[$d]}"
    if [[ "$d" == "full" && "$1" == "180" && "$CHAIN_HEADLINE" == "1" ]]; then links=3; epochs=30; fi
    chain "030-sclw1-N$1L$2-D$d-s1" "$links" "$HOURS" $(size "$1" "$2") $(sub "$d") +seed=1 "trainer.max_epochs=$epochs"
  done
done
# seeds 2 and 3 at the two smallest subsets, the two end sizes
for d in 64 16; do
  for wd in "45 2" "180 8"; do
    set -- $wd
    for s in 2 3; do
      chain "030-sclw1-N$1L$2-D$d-s$s" 1 "$HOURS" $(size "$1" "$2") $(sub "$d") "+seed=$s" "trainer.max_epochs=${EPOCHS[$d]}"
    done
  done
done
# learning-rate check at the smallest size on 1/4
for lr in 1e-4 1e-3; do
  chain "030-sclw1-N45L2-D4-lr$lr-s1" 1 "$HOURS" $(size 45 2) $(sub 4) +seed=1 "trainer.max_epochs=${EPOCHS[4]}" "regression_task.optimizer.lr=$lr"
done
echo "grid submitted"
