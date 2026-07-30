#!/bin/bash
# experiments/019-simb-multimodal/scripts/igb_expr_wave5.slurm
# [[experiments.019-simb-multimodal.scripts.igb_expr_wave5]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/igb_expr_wave5
#
# IGB half of WAVE 5. Resources come from the sbatch COMMAND LINE (CLI beats in-file
# directives), because mmli and cabbi need different partitions, memory and packing:
#
#   # canary -- prove the container + env + data path before committing a wave. ~5 min.
#   sbatch -p cabbi --gres=gpu:1 --cpus-per-task=8 --mem=60g --time=00:30:00 \
#       --array=0 --export=ALL,W5_STAGE=canary \
#       experiments/019-simb-multimodal/scripts/igb_expr_wave5.slurm
#
#   # H2 on mmli -- 4 x A100 80GB, 3 runs per GPU, 36 runs over 3 rounds.
#   sbatch -p mmli --gres=gpu:1 --cpus-per-task=12 --mem=100g --time=20:00:00 \
#       --array=0-11%4 --export=ALL,W5_STAGE=h2 \
#       experiments/019-simb-multimodal/scripts/igb_expr_wave5.slurm
#
#   # probes on cabbi -- 2 free RTX6000, ONE run per GPU, 8 runs over 4 rounds.
#   sbatch -p cabbi --gres=gpu:1 --cpus-per-task=8 --mem=60g --time=20:00:00 \
#       --array=0-7%2 --export=ALL,W5_STAGE=probe \
#       experiments/019-simb-multimodal/scripts/igb_expr_wave5.slurm
#
# PACKING DIFFERS BY PARTITION BECAUSE VRAM DOES, and this is not a tuning choice.
# One run measured 20,026 MiB. An A100 80GB therefore takes 3 (60 GB, ~20 GB headroom); a
# cabbi RTX6000 has 24 GB and takes exactly ONE -- packing two there would OOM. Packing is
# uniform WITHIN a stage, which is what the protocol requires: a paired reference is only a
# valid denominator for arms that ran beside it under identical conditions. It is NOT uniform
# ACROSS stages, so an mmli arm is never compared to a cabbi arm. Each stage co-launches its
# own W_ref for exactly this reason (_007 pooled A100 with RTX 6000 Ada runs; means 0.0242
# vs 0.0182).
#
# NO REQUEUE CHAIN, DELIBERATELY. The _007 launchers derived `--time` from a hardcoded
# DEADLINE minus 120 min of slack and also passed `--deadline`; when a job then waited in
# queue longer than the slack, its TimeLimit no longer fit and Slurm killed it instantly in
# DEADLINE state at 0 s elapsed, silently -- that is what happened to 2323142. Both
# partitions are MaxTime=UNLIMITED, so a plain fixed walltime removes the whole failure mode.
#
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --job-name=019-wave5-igb
#SBATCH --signal=B:USR1@180
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/home/a-m/mjvolk3/scratch/torchcell/experiments/019-simb-multimodal/slurm/output/%x_%A_%a.out
#SBATCH --error=/home/a-m/mjvolk3/scratch/torchcell/experiments/019-simb-multimodal/slurm/output/%x_%A_%a.out

set -euo pipefail

OUT_DIR=/home/a-m/mjvolk3/scratch/torchcell/experiments/019-simb-multimodal/slurm/output
mkdir -p "$OUT_DIR"

# The BRANCH worktree, not the main checkout. `torchcell` is pip-installed editable against
# /home/a-m/mjvolk3/projects/torchcell/torchcell, so without PYTHONPATH pointing here the
# import would silently resolve to the OTHER branch's source
# (memory: worktree-build-needs-pythonpath).
PROJECT_ROOT="${W5_PROJECT_ROOT:-/home/a-m/mjvolk3/projects/torchcell.worktrees/019-decoder-pair-routing}"
# The container lives in the main checkout and is untracked, so it is referenced by absolute
# path rather than expected inside the worktree.
SIF="${W5_SIF:-/home/a-m/mjvolk3/projects/torchcell/rockylinux_9.sif}"
ARM_SH="$PROJECT_ROOT/experiments/019-simb-multimodal/scripts/gh_expr_008_arm.sh"

STAGE="${W5_STAGE:?set W5_STAGE=canary|h2|probe via --export=ALL,W5_STAGE=...}"

# SEEDS ARE INIT-ONLY. cgt_expr_011 pins data_module.split_seed=0, so every run reads the
# SAME index_seed_0.json and these vary weight initialization only. That also removes the old
# constraint that seeds be drawn from {0,1,2,42}: concurrent tasks can no longer race to
# build a split cache, because there is only ever one split.
case "$STAGE" in
  # H2 -- does mixing genes AFTER the perturbation tell other genes how to express? The
  # encoder runs on the wildtype graph at batch 1, so H_genes is strain-independent and
  # nothing routes between gene tokens once the perturbation is applied. 9 paired replicates
  # per arm, because the prior evidence (E0_perceiver_on -0.0302, GEARS_crossgene -0.0243)
  # was n=1 on seed 0 alone and carries no significance.
  h2)     ARMS=(W_ref X_mix X_mix64 X_mix_rezero); SEEDS=(0 1 2 3 4 5 6 7 8)
          RUNS_PER_GPU=3
          EXTRA_OV="trainer.max_epochs=400 trainer.train_eval_every=5" ;;
  # PROBES -- lr below the frozen 3e-4 (the _007 top 12 cluster at 1e-4 while the wave-3
  # ladder only swept 3e-4/1e-3/3e-3), and dropout off on a model that underfits.
  probe)  ARMS=(W_ref P_lr1e4 P_lr3e5 H1_nodrop); SEEDS=(0 1)
          RUNS_PER_GPU=1
          EXTRA_OV="trainer.max_epochs=400 trainer.train_eval_every=5" ;;
  # CANARY -- one fast_dev_run to prove container + conda + PYTHONPATH + LMDB path before a
  # wave is committed to the queue. Exits in minutes.
  canary) ARMS=(X_mix); SEEDS=(0)
          RUNS_PER_GPU=1
          EXTRA_OV="trainer.fast_dev_run=true" ;;
  *) echo "unknown W5_STAGE '$STAGE' (expected canary|h2|probe)" >&2; exit 1 ;;
esac

# SEED-MAJOR, so each task's co-resident runs are DIFFERENT arms rather than three copies of
# one. The metric is unaffected either way (the budget is a fixed epoch count, so a slower
# neighbour costs wall time, not score), but arm-blocked packing aligns "what you share a GPU
# with" perfectly with the arm axis, and there is no reason to leave that correlation in.
JOBS=()
for seed in "${SEEDS[@]}"; do
  for arm in "${ARMS[@]}"; do
    JOBS+=("$arm:$seed")
  done
done

TASK="${SLURM_ARRAY_TASK_ID:?this script must be submitted with --array}"
START=$((TASK * RUNS_PER_GPU))
if [[ "$START" -ge "${#JOBS[@]}" ]]; then
  echo "task $TASK starts past the end of stage '$STAGE' (${#JOBS[@]} runs)" >&2
  exit 1
fi

# Reserve 2 CPUs for overhead, then split the rest across co-resident runs. Giving each the
# full allocation oversubscribes the cores and makes the interference non-uniform across tasks.
NUM_WORKERS=$(((SLURM_CPUS_PER_TASK - 2) / RUNS_PER_GPU))
[[ "$NUM_WORKERS" -lt 1 ]] && NUM_WORKERS=1

echo "=========================================================="
echo "host          : $(hostname)"
echo "partition     : ${SLURM_JOB_PARTITION:-unknown}"
echo "job           : ${SLURM_ARRAY_JOB_ID:-none} task=${TASK}"
echo "stage         : $STAGE  (runs ${START}..$((START + RUNS_PER_GPU - 1)) of ${#JOBS[@]})"
echo "project root  : $PROJECT_ROOT"
echo "gpus          : ${CUDA_VISIBLE_DEVICES:-unset}  ($RUNS_PER_GPU runs co-resident)"
echo "num_workers   : $NUM_WORKERS  (of $SLURM_CPUS_PER_TASK cpus)"
echo "=========================================================="

# `set -u` MUST be relaxed around the module/bashrc bootstrap. IGB's /etc/bashrc line 12
# references BASHRCSOURCED before defining it, so under `set -u` sourcing it aborts the job
# with "BASHRCSOURCED: unbound variable" -- which is exactly how the first canary (2324269)
# died, before reaching python at all. Cluster init scripts are not ours to make -u clean, so
# the strictness is dropped for the bootstrap and restored immediately after.
set +u
module purge
source ~/.bashrc
module load singularity
set -u

# ONE `singularity exec` PER RUN, not one container wrapping the whole loop. The loop lives in
# the OUTER shell so every variable (arm, seed, paths, worker count) is expanded here and the
# container receives a fully-formed command -- a loop nested inside `bash -c "..."` has to
# escape a bash array through several quoting layers, which is how a launcher silently runs the
# wrong arm. Concurrent singularity execs on one node are just processes and share the GPU
# exactly as concurrent python processes would.
#
# WANDB IS OFFLINE: compute nodes have no internet. Runs land under
# $DATA_ROOT/wandb-experiments/<group>/wandb/offline-run-* and are synced from the LOGIN node
# afterwards with scripts/wandb_sync_offline.sh (memory: cluster-rules-igb-delta).
CONTAINER_ENV="source \$HOME/miniconda3/bin/activate && conda activate torchcell"
CONTAINER_ENV="$CONTAINER_ENV && export PYTHONPATH='$PROJECT_ROOT'"
CONTAINER_ENV="$CONTAINER_ENV && export WANDB_MODE=offline WANDB__SERVICE_WAIT=600"
CONTAINER_ENV="$CONTAINER_ENV && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
CONTAINER_ENV="$CONTAINER_ENV && export TORCHCELL_CONFIG=cgt_expr_011"

PIDS=()
LABELS=()
for k in $(seq 0 $((RUNS_PER_GPU - 1))); do
  IDX=$((START + k))
  [[ "$IDX" -ge "${#JOBS[@]}" ]] && continue
  JOB="${JOBS[$IDX]}"
  ARM="${JOB%%:*}"
  SEED="${JOB##*:}"
  LOG="$OUT_DIR/019-wave5-igb_${SLURM_ARRAY_JOB_ID}_${TASK}_${ARM}_seed${SEED}.out"
  echo "  launching $ARM seed=$SEED -> $LOG"
  singularity exec --nv "$SIF" bash -c "
    # -u omitted on purpose: conda's activate scripts reference unset variables, the same
    # class of failure as /etc/bashrc above. -e and pipefail still catch real errors.
    set -eo pipefail
    $CONTAINER_ENV
    export TORCHCELL_EXTRA_OVERRIDES='$EXTRA_OV trainer.early_stopping.enabled=false data_module.num_workers=$NUM_WORKERS'
    echo \"python: \$(which python)\"
    echo \"overrides: \$TORCHCELL_EXTRA_OVERRIDES\"
    bash '$ARM_SH' '$ARM' '$SEED'
  " > "$LOG" 2>&1 &
  PIDS+=("$!")
  LABELS+=("$ARM:$SEED")
done

# Fail the task if ANY co-resident run fails: a half-completed task leaves an arm missing a
# replicate, which breaks the pairing for that replicate.
STATUS=0
for i in "${!PIDS[@]}"; do
  if wait "${PIDS[$i]}"; then
    echo "  OK     ${LABELS[$i]}"
  else
    echo "  FAILED ${LABELS[$i]}" >&2
    STATUS=1
  fi
done
exit "$STATUS"
