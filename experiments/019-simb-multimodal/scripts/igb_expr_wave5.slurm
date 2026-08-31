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
#   # H2 on mmli -- 4 x A100 80GB, 2 runs per GPU, 24 runs over 3 rounds.
#   # --cpus-per-task=16 is load-bearing: it is what gives 7 dataloader workers per run.
#   sbatch -p mmli --gres=gpu:1 --cpus-per-task=16 --mem=100g --time=20:00:00 \
#       --array=0-11%4 --export=ALL,W5_STAGE=h2 \
#       experiments/019-simb-multimodal/scripts/igb_expr_wave5.slurm
#
#   # probes on cabbi -- 2 free RTX6000, ONE run per GPU, 8 runs over 4 rounds.
#   sbatch -p cabbi --gres=gpu:1 --cpus-per-task=8 --mem=60g --time=20:00:00 \
#       --array=0-7%2 --export=ALL,W5_STAGE=probe \
#       experiments/019-simb-multimodal/scripts/igb_expr_wave5.slurm
#
# PACKING DIFFERS BY PARTITION BECAUSE VRAM DOES, and this is not a tuning choice.
# One run measured 20,026 MiB. An A100 80GB therefore takes 3 (60 GB, ~20 GB headroom).
#
# CORRECTION 2026-08-31: the cabbi cards are RTX 6000 **Ada**, 48 GB, not 24 GB -- read off
# W&B device metadata on compute-3-3 ("memory_total": 48305799168). The earlier claim that
# cabbi "takes exactly ONE -- packing two there would OOM" was wrong, and it halved the
# usable throughput of the larger partition. Two runs fit at 20 GB each with 8 GB headroom.
# Packing is still a per-stage decision rather than an automatic win: a campaign that needs
# LONG individual runs (the expression curve does not turn before ~10,000 epochs) is better
# served by one run per GPU, since packing shortens every run to raise aggregate throughput.
# Packing is
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

# PROJECT_ROOT resolves from THIS SCRIPT'S OWN LOCATION, so the code that runs is always the
# checkout the submitted script came from. It used to default to a named branch worktree,
# which meant submitting the script from anywhere else silently trained a different branch's
# source -- an un-attributable run that looks completely normal in the log.
#
# `torchcell` is pip-installed editable against /home/a-m/mjvolk3/projects/torchcell/torchcell,
# so PYTHONPATH must still point at PROJECT_ROOT or the import resolves to whatever branch
# that checkout happens to be on (memory: worktree-build-needs-pythonpath).
# W5_PROJECT_ROOT still overrides, for the deliberate case of running one checkout's script
# against another's source.
_W5_SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${W5_PROJECT_ROOT:-$(cd "$_W5_SELF/../../.." && pwd)}"
if [ ! -d "$PROJECT_ROOT/torchcell" ]; then
  echo "FATAL: PROJECT_ROOT=$PROJECT_ROOT has no torchcell/ package dir." >&2
  echo "       Submit this script from inside a torchcell checkout, or set W5_PROJECT_ROOT." >&2
  exit 1
fi
# The container lives in the main checkout and is untracked, so it is referenced by absolute
# path rather than expected inside the worktree.
SIF="${W5_SIF:-/home/a-m/mjvolk3/projects/torchcell/rockylinux_9.sif}"
ARM_SH="$PROJECT_ROOT/experiments/019-simb-multimodal/scripts/gh_expr_008_arm.sh"

STAGE="${W5_STAGE:?set W5_STAGE=canary|h2|probe|long|wave6 via --export=ALL,W5_STAGE=...}"

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
  # PACKING IS 2, NOT 3, AND THAT IS A MEASURED CORRECTION. The first mmli submission
  # (2324287) ran 3 runs/GPU with --cpus-per-task=12, giving each run only (12-2)/3 = 3
  # dataloader workers against GilaHyper's 7. After 14 minutes all 12 runs were at epoch 1-2
  # while GH reached ~23 and cabbi (1 run/GPU, 6 workers) reached 5 -- and the A100s read 0%
  # utilization at 25 GB, i.e. idle waiting on data rather than compute-bound. So the limit was
  # never VRAM (3 x 20 GB fits an 80 GB card comfortably); it was worker starvation, made worse
  # by LMDB reads over GPFS. At that rate 400 epochs x 3 rounds needed ~50 h against a 20 h
  # walltime. Submit with --cpus-per-task=16 so this yields (16-2)/2 = 7 workers, matching GH.
  #
  # X_mix64 IS DROPPED to buy the replicates back. It was a capacity variant of X_mix, whereas
  # W_ref / X_mix / X_mix_rezero are the actual H2 question: is the channel useful, and does
  # "not useful" separate from "never opened" (rezero starts it closed). 3 arms x 8 replicates
  # = 24 runs at 2/GPU = 3 rounds, which fits.
  h2)     ARMS=(W_ref X_mix X_mix_rezero); SEEDS=(0 1 2 3 4 5 6 7)
          RUNS_PER_GPU=2
          EXTRA_OV="trainer.max_epochs=300 trainer.train_eval_every=5" ;;
  # PROBES -- lr below the frozen 3e-4 (the _007 top 12 cluster at 1e-4 while the wave-3
  # ladder only swept 3e-4/1e-3/3e-3), and dropout off on a model that underfits.
  probe)  ARMS=(W_ref P_lr1e4 P_lr3e5 H1_nodrop); SEEDS=(0 1)
          RUNS_PER_GPU=1
          EXTRA_OV="trainer.max_epochs=400 trainer.train_eval_every=5" ;;
  # LONG -- the H1 question taken to its limit on ONE A100. The 1500-epoch GH runs showed
  # (a) eval-mode TRAIN pearson still rising at 0.64-0.72 with max == last in 4 of 4 runs, and
  # (b) val pearson dipping around epoch 200-300 and then climbing past its early peak to a
  # project-best 0.1980 at epoch 1367. Neither had plateaued, so the cap was still the binding
  # constraint rather than the model. 10000 epochs is deliberately far past where anything is
  # expected to move; it can be cancelled the moment the curves flatten or turn over, and
  # ModelCheckpoint now keeps best-by-loss AND best-by-metric AND last, so cancelling costs
  # nothing.
  #
  # Reference for reading the train side: the replicate-based expression ceiling is 0.775
  # (mean_g sqrt(rho_g), notes/experiments.019-simb-multimodal.scripts.expression_ceiling_replicate).
  # traineval already sits at 0.64-0.72, i.e. the model is approaching the point where it fits
  # TRAIN as well as the experiment reproduces itself -- past that, additional train fit is
  # fitting measurement noise by construction.
  #
  # train_eval_every=10 rather than 5: the eval-mode pass costs ~65 s on its own (it sweeps
  # 4074 train records against validation's 534), so at every=5 it would add ~13 s to every
  # epoch and cost ~36 h across 10000 of them. At every=10 that halves and still yields 1000
  # curve points.
  long)   ARMS=(W_ref); SEEDS=(0 1)
          RUNS_PER_GPU=2
          EXTRA_OV="trainer.max_epochs=10000 trainer.train_eval_every=10" ;;
  # WAVE 6 -- the pair-term rank ladder plus regularization, 12 arms, ONE seed, no replicates.
  # 3 runs/GPU x 4 A100 = 12 slots. Measured 23.51 s/epoch at this packing and batch 32
  # (jobs 1431-1434), so a 48 h walltime reaches ~7,350 of the 10,000-epoch cap -- the runs
  # are EXPECTED to hit the wall, which is fine because ModelCheckpoint retains best-by-loss,
  # best-by-metric and last. The noise floor is carried from wave 5's measured across-init
  # sd of ~0.006 (n=8, fixed split) rather than spent on in-wave replicates.
  wave6)  ARMS=(V_ref V_sink V_basis16 V_basis32 V_basis64 V_film V_hadamard V_hadamard_add
                V_drop2 V_drop3 V_wd1e4 V_wd1e2); SEEDS=(0)
          RUNS_PER_GPU=3; STAGE_CONFIG=cgt_expr_012
          EXTRA_OV="" ;;
  # CANARY -- one fast_dev_run to prove container + conda + PYTHONPATH + LMDB path before a
  # wave is committed to the queue. Exits in minutes.
  canary) ARMS=(X_mix); SEEDS=(0)
          RUNS_PER_GPU=1
          EXTRA_OV="trainer.fast_dev_run=true" ;;
  *) echo "unknown W5_STAGE '$STAGE' (expected canary|h2|probe|long|wave6)" >&2; exit 1 ;;
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
CONTAINER_ENV="$CONTAINER_ENV && export TORCHCELL_CONFIG=${STAGE_CONFIG:-cgt_expr_011}"
# SOURCE PROVENANCE MUST BE RESOLVED AT SUBMIT TIME, ON THE LOGIN NODE.
#
# Two canaries died establishing where git actually lives here. 2324270: the train script
# shelled out to `git rev-parse` and rockylinux_9.sif has no git -> FileNotFoundError('git').
# 2324272: moving that call into this launcher failed too, because **IGB COMPUTE NODES HAVE NO
# GIT EITHER** -- verified directly, `which git` on compute-3-3 returns nothing while
# /usr/bin/sha256sum is present. Only the login node has it.
#
# So the submitting shell exports the values and they ride in through --export:
#
#   H=$(git -C "$WT" rev-parse HEAD)
#   D=$(git -C "$WT" diff HEAD | sha256sum | cut -d" " -f1)
#   sbatch ... --export=ALL,W5_STAGE=h2,TORCHCELL_SOURCE_GIT_HASH=$H,TORCHCELL_SOURCE_DIFF_SHA256=$D ...
#
# The `command -v git` branch below is kept so the same script still self-resolves on a host
# that DOES have git (GilaHyper). If neither path yields a hash the run records "unavailable"
# and says so loudly -- an unattributable run must not be pooled with attributable ones.
SRC_HASH="${TORCHCELL_SOURCE_GIT_HASH:-}"
SRC_DIFF="${TORCHCELL_SOURCE_DIFF_SHA256:-}"
if [[ -z "$SRC_HASH" ]] && command -v git >/dev/null 2>&1; then
  SRC_HASH="$(git -C "$PROJECT_ROOT" rev-parse HEAD)"
  SRC_DIFF="$(git -C "$PROJECT_ROOT" diff HEAD | sha256sum | cut -d" " -f1)"
fi
if [[ -n "$SRC_HASH" && -n "$SRC_DIFF" ]]; then
  echo "source        : $SRC_HASH (diff ${SRC_DIFF:0:12})"
  CONTAINER_ENV="$CONTAINER_ENV && export TORCHCELL_SOURCE_GIT_HASH='$SRC_HASH'"
  CONTAINER_ENV="$CONTAINER_ENV && export TORCHCELL_SOURCE_DIFF_SHA256='$SRC_DIFF'"
else
  echo "source        : UNAVAILABLE -- no git here and no hash exported at submit time"
fi

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
