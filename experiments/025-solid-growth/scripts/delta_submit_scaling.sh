#!/bin/bash
# experiments/025-solid-growth/scripts/delta_submit_scaling.sh
# [[experiments.025-solid-growth.scripts.delta_submit_scaling]]
#
# The 025 half of the paired data curve: the SAME nested S3Q subsets as the 030 grid
# (delta_submit_scaling_030.sh), drawn by the same query pairs (subset_s3q_fractions.py),
# trained with ONE merged label per genotype by the 025 trainer on the composite closure
# arm of the Q split, cgt_s3_q_kl_embfit_037 (the representation of the 030 arm without the
# token). Scored on the same pinned 37,705 validation triples by MSE in label units, so the
# vertical gap to the 030 curve at every D is what keeping every stored measurement buys.
# One GPU per job, current size only (180x8), account bgcg-delta-gpu.
#
# PACE, corrected 2026-10-11: one A40 processes about 68 records/s (4-GPU 025 runs: 377k
# records in 23 min), so the full S3Q epoch is about 4.3 h and a 48 h job holds about 11;
# the first version of this file assumed 64 min per epoch, 4x too fast. Budgets match the
# 030 grid: 1/64 240 epochs, 1/16 120, 1/4 40, full 10, every point in ONE job (a dependent
# link accrues no queue priority, so chains pay a queue wait per link); the full point is
# the one chain, 3 links of 48 h for 30 epochs, resuming the full training state
# (RESUME_FROM_JOB in delta_cgt.slurm, afterany). CHAIN_HEADLINE=0 makes it one job.
# Each job stages the 554 GB build to node-local NVMe (about 30 min), as every 025 Delta
# job does; two staged jobs fit on one node's local disk, a third fails fast and is
# resubmitted. Batch 256 on one GPU, as the 030 grid.
#
#   RUN ON A DELTA LOGIN NODE, FROM THE WORKTREE ROOT, after delta_preflight_025.sh.
#   DRY=1 bash experiments/025-solid-growth/scripts/delta_submit_scaling.sh
#         bash experiments/025-solid-growth/scripts/delta_submit_scaling.sh
set -euo pipefail
ACCOUNT="${ACCOUNT:-bgcg-delta-gpu}"
LAUNCHER="experiments/025-solid-growth/scripts/delta_cgt.slurm"
CONFIG="cgt_s3_q_kl_embfit_037"
SWEEP="${SWEEP:-025_sclw1}"
HOURS="${HOURS:-48}"
MEM="${MEM:-100g}"
CPUS="${CPUS:-16}"
BATCH="${BATCH:-256}"
DRY="${DRY:-0}"
[[ -f "$LAUNCHER" ]] || { echo "run from the worktree root: $LAUNCHER not found" >&2; exit 2; }
[[ -f "experiments/025-solid-growth/conf/$CONFIG.yaml" ]] || { echo "missing $CONFIG" >&2; exit 2; }
for f in subset_S3Q_indices subset_S3Q_frac4_indices subset_S3Q_frac16_indices subset_S3Q_frac64_indices; do
  [[ -f "experiments/025-solid-growth/results/$f.json.gz" ]] || { echo "missing results/$f.json.gz" >&2; exit 2; }
done
COMMON=(trainer.devices=1 "data_module.batch_size=$BATCH" "+scaling.sweep=$SWEEP")

submit() {  # submit <name> <hours> <dependency-or-empty> [overrides...]; prints the job id
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
chain() {  # chain <name> <links> [overrides...]
  local name="$1" links="$2"; shift 2
  local prev="" id
  for ((k = 1; k <= links; k++)); do
    id=$(submit "$name-c$k" "$HOURS" "$prev" "$@"); echo "  -> $id" >&2; prev="$id"
  done
}
sub() { if [[ "$1" == "full" ]]; then echo "subset.indices=subset_S3Q_indices.json.gz"; else echo "subset.indices=subset_S3Q_frac$1_indices.json.gz"; fi; }
declare -A EPOCHS=([64]=240 [16]=120 [4]=40 [full]=10)
declare -A LINKS=([64]=1 [16]=1 [4]=1 [full]=1)
CHAIN_HEADLINE="${CHAIN_HEADLINE:-0}"
if [[ "$CHAIN_HEADLINE" == "1" ]]; then LINKS[full]=3; EPOCHS[full]=30; fi
for d in 64 16 4 full; do
  chain "025-sclw1-N180L8-D$d-s1" "${LINKS[$d]}" $(sub "$d") +seed=1 "trainer.max_epochs=${EPOCHS[$d]}"
done
for d in 64 16; do
  for s in 2 3; do
    chain "025-sclw1-N180L8-D$d-s$s" 1 $(sub "$d") "+seed=$s" "trainer.max_epochs=${EPOCHS[$d]}"
  done
done
echo "025 paired curve submitted"
