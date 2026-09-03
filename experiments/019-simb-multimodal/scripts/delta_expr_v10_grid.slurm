#!/bin/bash
# experiments/019-simb-multimodal/scripts/delta_expr_v10_grid.slurm
# [[experiments.019-simb-multimodal.scripts.delta_expr_v10_grid]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/delta_expr_v10_grid
#
# THE GENERALIZATION-GAP ROUND on Delta: a 2^4 factorial, 16 cells x 4 replicates = 64 runs,
# two runs per A40 across 32 single-GPU array tasks, which Slurm packs onto 8 nodes.
#
#   # PRE-FLIGHT, four corner cells at fast_dev_run. Minutes. Run this FIRST.
#   sbatch --array=0-3 --export=ALL,V10_STAGE=canary \
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
# ACCOUNT. `bbtp-delta-gpu` (CHM230022) is what the older delta_joint_expr_morph.slurm names
# and it is NO LONGER in this user's associations -- `sacctmgr show assoc` lists bbhh, bbub,
# bfjt, bflt and bgcg only, so a job charging bbtp is rejected at submit. `bbub-delta-gpu` is
# used here because storage already lives on bbub. bbhh was overdrawn when last checked.
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
#SBATCH --account=bbub-delta-gpu
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

STAGE="${V10_STAGE:?set V10_STAGE=canary|grid via --export=ALL,V10_STAGE=...}"
STAGE_CONFIG=cgt_expr_v10_grid

# ---------------------------------------------------------------- THE GRID
# Sixteen cells, indexed by a 4-BIT NUMBER so the mapping is arithmetic rather than a table
# that can drift out of sync with the arm list. Bit set means the second level.
#
#   bit 0  embedding     0 random_1024        1 prot_T5_all
#   bit 1  trunk         0 L=6  hidden=90     1 L=2  hidden=45
#   bit 2  readout       0 two-layer MLP      1 linear
#   bit 3  weight decay  0 1e-8               1 1e-2
#
# WHY THESE FOUR. The model fits TRAIN to 0.786 against a 0.775 replicate ceiling while
# validating at 0.240, so it is memorizing rather than underfitting, and the gap is ~0.55
# while every architecture arm tried so far moves 0.01-0.05. These are the four levers that
# act on the gap. Weight decay is currently 1e-8, which is numerically off, and has never
# been measured on this model at any replicate count.
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
WD=(1e-8 1e-2)

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
  # THE ROUND. 16 cells x 4 replicates = 64 runs over 8 array tasks, each a whole node
  # running 8 runs as TWO sequential rounds of four, one run per GPU.
  grid)   CELLS=($(seq 0 15)); REPS=4; ROUNDS=2
          EXTRA_OV="" ;;
  *) echo "unknown V10_STAGE '$STAGE' (expected canary|grid)" >&2; exit 1 ;;
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

GPUS_PER_NODE=4
PER_TASK=$((GPUS_PER_NODE * ROUNDS))

TASK="${SLURM_ARRAY_TASK_ID:?this script must be submitted with --array}"
START=$((TASK * PER_TASK))
if [[ "$START" -ge "${#JOBS[@]}" ]]; then
  echo "task $TASK starts past the end of stage '$STAGE' (${#JOBS[@]} runs)" >&2
  exit 1
fi

# Reserve 2 CPUs for overhead and split the rest across the four CONCURRENT runs, giving 7
# workers each at 32 CPUs. This is the same worker count every healthy run in this campaign
# has used, and staying at it is the whole reason for two sequential rounds rather than
# eight concurrent runs on one node.
NUM_WORKERS=$(((SLURM_CPUS_PER_TASK - 2) / GPUS_PER_NODE))
[[ "$NUM_WORKERS" -lt 1 ]] && NUM_WORKERS=1

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
export DATA_ROOT=/work/hdd/bbub/mjvolk3/torchcell
export PYTHONPATH="$PROJECT_ROOT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB__SERVICE_WAIT=600
cd "$PROJECT_ROOT" || exit 1

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
