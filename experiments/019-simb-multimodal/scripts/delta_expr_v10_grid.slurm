#!/bin/bash
# experiments/019-simb-multimodal/scripts/delta_expr_v10_grid.slurm
# [[experiments.019-simb-multimodal.scripts.delta_expr_v10_grid]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/delta_expr_v10_grid
#
# THE GENERALIZATION-GAP ROUND on Delta: a 2^4 factorial, 16 cells x 4 replicates = 64 runs,
# two runs per A40 across 32 single-GPU array tasks, which Slurm packs onto 8 nodes.
#
#   # PRE-FLIGHT, four corner cells at fast_dev_run. Minutes of COMPUTE, but see the wall.
#   # OVERRIDE THE WALLTIME DOWN, BUT NOT TO AN HOUR. The #SBATCH block asks 2 days because
#   # the grid needs it, and a 2-day request cannot be backfilled, so a canary inheriting it
#   # queues behind full-size work. But STARTUP DOMINATES A CANARY HERE: dataset load plus
#   # target normalization was measured at ~43 min on Delta once, and canary 21785783 hit a
#   # 1 h wall with all four runs still inside startup. 3 h leaves room and still backfills.
#   sbatch --array=0 --time=03:00:00 --export=ALL,V10_STAGE=canary \
#       experiments/019-simb-multimodal/scripts/delta_expr_v10_grid.slurm
#
#   # THE ROUND.
#   sbatch --array=0-31 --export=ALL,V10_STAGE=grid \
#       experiments/019-simb-multimodal/scripts/delta_expr_v10_grid.slurm
#
# ONE-TIME SETUP: Slurm opens --output BEFORE this script's own mkdir runs, so the log
# directory must already exist or every task fails in 0 s with no log at all.
#   mkdir -p /projects/bbub/mjvolk3/torchcell/experiments/019-simb-multimodal/slurm/output
#
# ACCOUNT. `bbtp-delta-gpu` (CHM230022), which the older delta_joint_expr_morph.slurm names,
# is NO LONGER in this user's associations, and a submit under it returns
# "allocation failure: Invalid account or account/partition combination" -- verified, not
# assumed. Balances as of 2026-09-03: bfjt 12,843 h, bflt 4,845, bgcg 4,760, bbub 3,851,
# bbhh OVERDRAWN at -6,447. `bfjt-delta-gpu` is used because it has by far the most headroom
# and this round wants ~1,000 GPU-hours. STORAGE stays on bbub regardless; only the charge
# account differs, which is the same split the earlier Delta launchers used.
#
# RESOURCE SHAPE COPIED FROM WHAT DELTA HAS RECENTLY ACCEPTED, not chosen fresh. Jobs
# 20714233-20714235 (020/021/022/023 grids) each COMPLETED at 1-23:53 with exactly
# `--nodes=1 --cpus-per-task=32 --gpus-per-node=4 --mem=180g --time=2-00:00:00` and no
# --constraint. A gpuA40x4 node therefore gives 32 CPUs to 4 GPUs, i.e. EIGHT CPUs per GPU.
# An earlier draft of this file asked for 16 CPUs alongside one GPU, which does not fit that
# ratio: it would have spread 32 tasks over sixteen nodes instead of eight, or pended.
# `--constraint="scratch"` is deliberately absent -- the older delta_joint template carries
# it, the recently successful jobs do not, and an unsatisfiable constraint pends silently.
#
# ONE RUN PER GPU, TWO ROUNDS, rather than two runs per GPU in one round. Packing eight runs
# onto a node's 32 CPUs leaves 3 dataloader workers each, and 3 workers is the measured
# starvation point: IGB submission 2324287 ran 3 runs/GPU at 3 workers and sat at epoch 1-2
# while an unpacked run reached 23, with the GPUs reading 0% utilization, i.e. idle waiting
# on data. Solo runs get 7 workers, which is the configuration every healthy run in this
# campaign has used. Two sequential rounds of four fit the wall: at the measured ~128
# epochs/h solo, 2,000 epochs is ~15.6 h, so two rounds plus startup is ~33 h against 48.
#
#SBATCH --job-name=019-expr-v10
#SBATCH --partition=gpuA40x4
#SBATCH --account=bfjt-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=180g
#SBATCH --time=2-00:00:00
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/projects/bbub/mjvolk3/torchcell/experiments/019-simb-multimodal/slurm/output/%x_%A_%a.out
#SBATCH --error=/projects/bbub/mjvolk3/torchcell/experiments/019-simb-multimodal/slurm/output/%x_%A_%a.out

set -euo pipefail

PROJECT_ROOT="${V10_PROJECT_ROOT:-/projects/bbub/mjvolk3/torchcell}"
OUT_DIR="$PROJECT_ROOT/experiments/019-simb-multimodal/slurm/output"
mkdir -p "$OUT_DIR"
if [ ! -d "$PROJECT_ROOT/torchcell" ]; then
  echo "FATAL: PROJECT_ROOT=$PROJECT_ROOT has no torchcell/ package dir." >&2
  exit 1
fi

STAGE="${V10_STAGE:?set V10_STAGE=canary|timing|grid via --export=ALL,V10_STAGE=...}"
STAGE_CONFIG=cgt_expr_v10_grid

# ---------------------------------------------------------------- THE GRID
# Sixteen cells, indexed by a 4-BIT NUMBER so the mapping is arithmetic rather than a table
# that can drift out of sync with the arm list. Bit set means the second level.
#
#   bit 0  embedding     0 random_1024        1 prot_T5_all
#   bit 1  trunk         0 L=6  hidden=90     1 L=2  hidden=45
#   bit 2  readout       0 two-layer MLP      1 linear
#   bit 3  weight decay  0 1e-8               1 1e-4
#
# WHY THESE FOUR. The model fits TRAIN to 0.786 against a 0.775 replicate ceiling while
# validating at 0.240, so it is memorizing rather than underfitting, and the gap is ~0.55
# while every architecture arm tried so far moves 0.01-0.05. These are the four levers that
# act on the gap. Weight decay is currently 1e-8, numerically off, never measured here.
#
# THE SECOND WEIGHT-DECAY LEVEL IS 1e-4, NOT 1e-2, AND THAT IS EVIDENCE-DRIVEN. Every 010
# config pins weight_decay 1e-8 with the note "transformer embeddings are sensitive to L2",
# and cgt_expr_007 records that 010's checkpointed model used 1e-8 against the naive
# reading. 1e-4 is the value _007's alternative bundle carried, so it is a level this
# architecture has actually seen. A jump to 1e-2 on an architecture documented as L2
# sensitive risks driving cells to a CONSTANT prediction, and a collapsed run scores exactly
# 0.000 and averages into its factor mean like a real value -- the v9 point head sat at
# pred_sd_ratio 0.000 through epoch 7,730 and never recovered, so this is a measured failure
# mode of this model, not a hypothetical.
#
# WHY random_1024. prot_T5 is 1024-dimensional, so a control of any other width changes the
# embedding preprocessor's input size alongside the content and leaves "content helps"
# confounded with "a wider preprocessor helps". At 1024 the two levels feed an IDENTICALLY
# SIZED preprocessor and the factor is content and nothing else. The name had to be added to
# the embedding registry to be reachable from a config; a 1024-wide tensor was already built
# on Delta but no config could name it. Both levels verified to load at 6,607 genes x 1024.
#
# HIDDEN 45, NOT 48, because the layer-1 attention mask assigns ONE GRAPH PER HEAD across
# nine graphs, so the head count is pinned at 9 and the width must divide by it. 45 gives 5
# per head against the incumbent's 10. Changing the head count instead would change the
# graph prior and confound the trunk factor with it.
EMB=(random_1024 prot_T5_all)
TRUNK_L=(6 2)
TRUNK_H=(90 45)
READOUT=(false true)
WD=(1e-8 1e-4)

cell_overrides() {  # $1 = cell index 0..15
  local c=$1
  local e=$((c % 2)) t=$(((c / 2) % 2)) r=$(((c / 4) % 2)) w=$(((c / 8) % 2))
  echo "cell_dataset.node_embeddings=[${EMB[$e]}]" \
       "model.num_transformer_layers=${TRUNK_L[$t]}" \
       "model.hidden_channels=${TRUNK_H[$t]}" \
       "multitask.linear_readout=${READOUT[$r]}" \
       "regression_task.optimizer.weight_decay=${WD[$w]}"
}

cell_name() {  # a short human-readable label for the log file and the W&B run
  local c=$1
  local e=$((c % 2)) t=$(((c / 2) % 2)) r=$(((c / 4) % 2)) w=$(((c / 8) % 2))
  local en=(rand ptt5) tn=(big small) rn=(mlp lin) wn=(wd0 wd2)
  echo "c${c}_${en[$e]}_${tn[$t]}_${rn[$r]}_${wn[$w]}"
}

case "$STAGE" in
  # PRE-FLIGHT. Four CORNER cells, chosen so that every level of every factor appears at
  # least once: cell 0 is all-first-level, cell 15 all-second, and 3 and 12 cross them.
  # fast_dev_run exercises the real dataset, the real embeddings and the real widths, which
  # the in-script dry_run cannot: its synthetic graph carries a 1-dimensional feature, so it
  # never builds the embedding preprocessor at all and cannot catch a width mismatch.
  canary) CELLS=(0 3 12 15); REPS=1; ROUNDS=1
          EXTRA_OV="trainer.fast_dev_run=true" ;;
  # TIMING. The same four corner cells, but REAL training for a few dozen epochs, because
  # fast_dev_run is one batch and says nothing about throughput.
  #
  # WHY THIS EXISTS AS ITS OWN STAGE. The grid's 2,000-epoch budget rests on ~128 epochs/h,
  # measured on an IGB A40 with SEVEN dataloader workers. Delta runs with ZERO, so batches
  # are assembled in the main process off Lustre rather than by seven prefetching workers on
  # local NVMe. Transplanting the IGB rate across that change is a guess, and if it is wrong
  # the round hits its wall mid-training, which is the exact failure the two-round structure
  # was built to avoid. Corner cells span the trunk sizes, so the SLOWEST of them is what
  # sizes the budget.
  #
  # train_eval_every matches the grid so the measured rate includes the eval-mode train pass
  # the real runs pay for, rather than a clean number the round will never see.
  timing) CELLS=(0 3 12 15); REPS=1; ROUNDS=1
          EXTRA_OV="trainer.max_epochs=60 trainer.train_eval_every=10" ;;
  # THE ROUND. 16 cells x 4 replicates = 64 runs over 8 array tasks, each a whole node
  # running 8 runs as TWO sequential rounds of four, one run per GPU.
  grid)   CELLS=($(seq 0 15)); REPS=4; ROUNDS=2
          EXTRA_OV="" ;;
  *) echo "unknown V10_STAGE '$STAGE' (expected canary|timing|grid)" >&2; exit 1 ;;
esac

# RUN ORDER = NODE ASSIGNMENT, so it is chosen to keep every factor balanced WITHIN a node
# rather than left as the natural 0..15 sequence.
#
# THE DEFECT IT AVOIDS. A node takes eight consecutive runs. In cell order 0..15 that hands
# node A cells 0-7, which are ALL weight-decay 1e-8, and node B cells 8-15, which are ALL
# 1e-2. Weight decay would then be perfectly confounded with which physical node ran it: a
# slow, hot or otherwise unlucky node would show up as a weight-decay effect and there would
# be no way to tell the two apart after the fact.
#
# THE FIX. Split the sixteen cells by the PARITY of their four bits. The even-parity half
# {0,3,5,6,9,10,12,15} contains exactly four cells at each level of each of the four factors,
# and so does its complement. Giving one node each half means a node-level effect lands
# equally on both levels of every factor and cancels out of every main effect. This is the
# standard resolution-IV half-fraction, used here for physical blocking rather than to
# shrink the design.
if [[ "$STAGE" == "grid" ]]; then
  EVEN=(); ODD=()
  for c in "${CELLS[@]}"; do
    par=$(( (c & 1) ^ ((c >> 1) & 1) ^ ((c >> 2) & 1) ^ ((c >> 3) & 1) ))
    if [[ "$par" -eq 0 ]]; then EVEN+=("$c"); else ODD+=("$c"); fi
  done
  JOBS=()
  for rep in $(seq 0 $((REPS - 1))); do
    for c in "${EVEN[@]}"; do JOBS+=("$c:$rep"); done
    for c in "${ODD[@]}";  do JOBS+=("$c:$rep"); done
  done
else
  JOBS=()
  for rep in $(seq 0 $((REPS - 1))); do
    for c in "${CELLS[@]}"; do
      JOBS+=("$c:$rep")
    done
  done
fi

# GPU COUNT COMES FROM THE ALLOCATION, not a constant. Slurm sets SLURM_GPUS_ON_NODE to
# what this task actually got, so the same script is correct whether it is handed a whole
# 4-GPU node for the grid or a single card for a short canary. Hardcoding 4 meant a canary
# submitted with --gpus-per-node=1 would have tried to pin runs to cards 1-3 that it does
# not own, and every one of those would have failed on an invalid device.
GPUS_PER_NODE="${SLURM_GPUS_ON_NODE:-4}"
PER_TASK=$((GPUS_PER_NODE * ROUNDS))

TASK="${SLURM_ARRAY_TASK_ID:?this script must be submitted with --array}"
START=$((TASK * PER_TASK))
if [[ "$START" -ge "${#JOBS[@]}" ]]; then
  echo "task $TASK starts past the end of stage '$STAGE' (${#JOBS[@]} runs)" >&2
  exit 1
fi

# DATALOADER WORKERS ARE 0 ON DELTA, AND THAT IS A MEASUREMENT, NOT A FALLBACK.
#
# The datamodule uses multiprocessing_context="spawn", which is required because CUDA is
# already initialized in the parent. Under spawn every worker is a FRESH interpreter that
# re-imports torch, PyG and torchcell from the parallel filesystem. On GilaHyper's local
# NVMe that costs seconds. On Delta's Lustre it does not: delta_grid_common.sh records an
# interactive smoke test that sat in Sanity Checking for 38 MINUTES without reaching one
# validation batch, same code and same dataset, and names the filesystem as the variable.
#
# THIS LAUNCHER ALREADY PAID FOR THAT LESSON TWICE. Canaries 21785783 and 21788244 both
# stalled with 7 workers per run: four python processes, 11 seconds of CPU each across 35
# minutes, every one blocked in `cl_sync_io_wait`, which is Lustre's client I/O wait. The
# IGB reasoning that produced 7 workers is correct FOR IGB, where the worry is starvation
# on a local filesystem, and exactly inverted here.
#
# 0 removes the mechanism: no spawn, load in the main process. The dataset is small enough
# that this is a real option. It also needs the prefetch_factor guard from 2cca6d83, without
# which num_workers=0 is not slow but UNCONSTRUCTIBLE -- that is what killed all 119 trials
# of Delta job 20556837 in ~15 s apiece while the job still exited COMPLETED 0:0.
NUM_WORKERS="${DELTA_NUM_WORKERS:-0}"

echo "=========================================================="
echo "host          : $(hostname)"
echo "partition     : ${SLURM_JOB_PARTITION:-unknown}  account: ${SLURM_JOB_ACCOUNT:-unknown}"
echo "job           : ${SLURM_ARRAY_JOB_ID:-none} task=${TASK}"
echo "stage         : $STAGE  (runs ${START}..$((START + PER_TASK - 1)) of ${#JOBS[@]}, $ROUNDS round(s) x $GPUS_PER_NODE GPUs)"
echo "project root  : $PROJECT_ROOT"
echo "config        : $STAGE_CONFIG"
echo "num_workers   : $NUM_WORKERS  (of $SLURM_CPUS_PER_TASK cpus, 1 run/GPU)"
echo "=========================================================="

# SOURCE PROVENANCE resolved at SUBMIT time on the login node, same as the IGB launcher: a
# compute node may not have git, and an unattributable run must not be pooled with
# attributable ones.
SRC_HASH="${TORCHCELL_SOURCE_GIT_HASH:-}"
SRC_DIFF="${TORCHCELL_SOURCE_DIFF_SHA256:-}"
if [[ -z "$SRC_HASH" ]] && command -v git >/dev/null 2>&1; then
  SRC_HASH="$(git -C "$PROJECT_ROOT" rev-parse HEAD)"
  SRC_DIFF="$(git -C "$PROJECT_ROOT" diff HEAD | sha256sum | cut -d" " -f1)"
fi
if [[ -n "$SRC_HASH" && -n "$SRC_DIFF" ]]; then
  echo "source        : $SRC_HASH (diff ${SRC_DIFF:0:12})"
else
  echo "source        : UNAVAILABLE -- no git here and no hash exported at submit time"
fi

# CALL THE ENV PYTHON DIRECTLY. /work/hdd/bbub/miniconda3 is a PREFIX location, not a conda
# install, so it has no bin/activate; `source .../bin/activate` there fails SILENTLY and the
# job then runs system python3.9. Delta compute nodes have outbound network, so W&B is
# ONLINE here and needs no login-node sync afterwards, unlike IGB.
ENV_PY=/work/hdd/bbub/miniconda3/envs/torchcell/bin/python
# DATA_ROOT is /scratch, NOT /work/hdd. Both currently hold a 13G fig3_core build, so this
# is not what stalled the canaries, but delta_grid_common.sh calls the inconsistency "the
# single most expensive thing here to get wrong": Neo4jCellDataset skips the database only
# when processed/ exists, so a root without a build makes a compute node try to REBUILD with
# no Neo4j. Following the confirmed root, and failing fast rather than discovering it twenty
# minutes into a two-day job.
export DATA_ROOT="${DELTA_DATA_ROOT:-/scratch/bbub/mjvolk3/torchcell}"
if [ ! -d "$DATA_ROOT/data/torchcell/experiments/019-simb-multimodal/fig3_core/processed" ]; then
  echo "FATAL: no built fig3_core under DATA_ROOT=$DATA_ROOT" >&2
  echo "       Set DELTA_DATA_ROOT at submit time; do not let the loader try to rebuild." >&2
  exit 1
fi
export PYTHONWARNINGS=ignore
export PYTHONPATH="$PROJECT_ROOT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB__SERVICE_WAIT=600
# UNBUFFERED, and this is not cosmetic. Python FULLY buffers stdout when it is redirected to
# a file, so a run that has printed less than one buffer's worth writes NOTHING. Canary
# 21785783 hit its wall with all four per-run logs at exactly 0 bytes, which is
# indistinguishable between "still loading the dataset" and "hung before the first line".
# Unbuffered output makes startup visible and separates those two cases.
export PYTHONUNBUFFERED=1
cd "$PROJECT_ROOT" || exit 1

# IMPORT PROVENANCE GATE. The Delta environment carries an EDITABLE install of torchcell,
# and an editable install works through a meta-path finder, which Python consults BEFORE
# anything on sys.path -- including PYTHONPATH. So if that finder ever points somewhere
# other than PROJECT_ROOT, every run here silently trains a different checkout's source
# while the log still prints this checkout's git hash: an unattributable result that looks
# completely normal. It points at PROJECT_ROOT today; this asserts it rather than trusting
# it, and turns a silent wrong-code run into an immediate failure before any GPU is spent.
RESOLVED="$("$ENV_PY" -c 'import torchcell, os; print(os.path.realpath(os.path.dirname(torchcell.__file__)))')"
EXPECTED="$(cd "$PROJECT_ROOT" && pwd -P)/torchcell"
if [[ "$RESOLVED" != "$EXPECTED" ]]; then
  echo "FATAL: torchcell imports from $RESOLVED but PROJECT_ROOT is $EXPECTED." >&2
  echo "       An editable install or a stale site-packages copy is shadowing the repo." >&2
  exit 1
fi
echo "import check  : torchcell -> $RESOLVED (matches PROJECT_ROOT)"

TRAIN=experiments/019-simb-multimodal/scripts/train_cgt_multitask.py
RC=0
for round in $(seq 0 $((ROUNDS - 1))); do
  PIDS=()
  LABELS=()
  for gpu in $(seq 0 $((GPUS_PER_NODE - 1))); do
    IDX=$((START + round * GPUS_PER_NODE + gpu))
    [[ "$IDX" -ge "${#JOBS[@]}" ]] && continue
    JOB="${JOBS[$IDX]}"
    CELL="${JOB%%:*}"
    REP="${JOB##*:}"
    NAME="$(cell_name "$CELL")"
    LOG="$OUT_DIR/019-expr-v10_${SLURM_ARRAY_JOB_ID}_${TASK}_${NAME}_seed${REP}.out"
    OV="$(cell_overrides "$CELL")"
    echo "  [round $round gpu $gpu] cell $CELL ($NAME) seed=$REP -> $LOG"
    echo "    overrides: $OV"
    # CUDA_VISIBLE_DEVICES pins ONE card per run. Without it every run would open device 0
    # and four processes would contend for one GPU while three sat idle, which looks like a
    # slow round rather than a misconfiguration.
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES="$gpu" "$ENV_PY" "$TRAIN" --config-name "$STAGE_CONFIG" \
        $OV $EXTRA_OV \
        "regression_task.optimizer.lr=3e-4" \
        "data_module.num_workers=$NUM_WORKERS" \
        "trainer.early_stopping.enabled=false" \
        "wandb.tags=[v10,$NAME,seed$REP,cell$CELL]" \
        "seed=$REP" \
        > "$LOG" 2>&1 &
    PIDS+=("$!")
    LABELS+=("$NAME:$REP")
    # STAGGER. Four interpreters importing the same multi-gigabyte stack off Lustre at the
    # same instant is the worst case for that filesystem, and it is what the canaries hit.
    # Once the first has read those files the node's client cache is warm and the rest are
    # far cheaper, so a short offset costs a minute and can save many.
    [[ "$gpu" -lt $((GPUS_PER_NODE - 1)) ]] && sleep "${DELTA_LAUNCH_STAGGER_S:-60}"
  done

  # Wait for the whole round before starting the next. A failure is RECORDED and the next
  # round still runs: one broken cell must not cost the four runs queued behind it, which is
  # what an early exit here would do.
  for i in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$i]}"; then
      echo "FAILED: round $round ${LABELS[$i]}" >&2
      RC=1
    fi
  done
  echo "  round $round complete"
done
exit "$RC"
