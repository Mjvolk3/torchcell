#!/bin/bash
# experiments/019-simb-multimodal/scripts/delta_expr_v10_grid.slurm
# [[experiments.019-simb-multimodal.scripts.delta_expr_v10_grid]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/delta_expr_v10_grid
#
# THE GENERALIZATION-GAP ROUND on Delta: a 2^4 factorial, 16 cells x 2 replicates = 32 runs,
# ONE run per A40 for the whole job, four per node, across 8 whole-node array tasks.
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
#   # V10_PROJECT_ROOT IS NOT OPTIONAL, and grid 21796128 is why. It ran from the SHARED
#   # checkout /projects/bbub/mjvolk3/torchcell, which was switched to another branch at
#   # 01:11 for unrelated 027 work while the job was live. Every one of the 64 runs died on
#   # "Cannot find primary config 'cgt_expr_v10_grid'", and the tasks still looked alive in
#   # squeue because they were sitting in the launch stagger loop. A multi-day job must run
#   # from a checkout it OWNS. The clone is made once with:
#   #   git clone --branch multimodal-phenotype-retrospective \
#   #       /projects/bbub/mjvolk3/torchcell /work/hdd/bbub/mjvolk3/torchcell-v10
#   # --account is passed explicitly so this round charges a DIFFERENT project from whatever
#   # else is queued. Two jobs on one account share a fairshare pool and hold each other back;
#   # on 2026-09-04 this grid and an unrelated 027 array both sat on bfjt and one read
#   # "(Priority)" purely because of the other. Balances that day: bfjt 12,788 h, bflt 4,845,
#   # bgcg 4,760, bbub 3,738, bbhh OVERDRAWN. This round wants ~1,500 GPU-hours.
#   sbatch --array=0-7 --account=bflt-delta-gpu \
#       --export=ALL,V10_STAGE=grid,V10_PROJECT_ROOT=/work/hdd/bbub/mjvolk3/torchcell-v10 \
#       /work/hdd/bbub/mjvolk3/torchcell-v10/experiments/019-simb-multimodal/scripts/delta_expr_v10_grid.slurm
#
# ONE-TIME SETUP: Slurm opens --output BEFORE this script's own mkdir runs, so the log
# directory must already exist or every task fails in 0 s with no log at all. See the
# --output directive below for why this path deliberately sits outside every git checkout.
#   mkdir -p /work/hdd/bbub/mjvolk3/slurm-logs/019-expr-v10
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
# ONE RUN PER GPU, ONE ROUND. Packing several runs onto a node's 32 CPUs is what this avoids:
# IGB submission 2324287 ran 3 runs/GPU and sat at epoch 1-2 while an unpacked run reached 23,
# with the GPUs reading 0% utilization, i.e. idle waiting on data. On Delta the packing
# argument is moot anyway, since dataloader workers are 0 here (spawn workers re-importing the
# stack off Lustre is worse than no workers at all, see delta_grid_common.sh).
#
# THE ROUND COUNT IS SET BY MEASUREMENT, and it changed. An earlier version ran TWO sequential
# rounds of four, sized against an assumed ~128 epochs/h carried over from IGB. The timing
# stage measured 33.6 epochs/h, at which two rounds afford only 700 epochs each. Extending
# short_budget_spread.py down to 350 showed 700 is the noisiest point on the budget grid
# (sd 0.0166, against 0.0058 at 500 and 0.0117 at 1,500) and that one long round of 32 runs
# resolves the same main effect, 0.0117, as two short rounds of 64. One round it is: 46.2 h of
# training after a 1.75 h startup is ~1,552 epochs, and the config asks for 1,400.
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
# LOGS LIVE OUTSIDE EVERY GIT CHECKOUT, and that is the whole point of the path. #SBATCH
# directives are STATIC: Slurm parses them at submit time and opens the output file BEFORE
# the script body runs, so they cannot read V10_PROJECT_ROOT and the script's own mkdir is
# far too late. Grid 21813317 pointed here at the SHARED checkout's slurm/output, which had
# ceased to exist when that checkout moved to another branch, and all 8 tasks died in 8
# seconds with exit 128 and NO LOG AT ALL, before any gate in this file could run. Pointing
# at the job-owned checkout instead would only narrow the window, since a clone is still
# something a git operation can rewrite. A plain directory under /work/hdd that no checkout
# owns cannot be removed by any branch switch, reset or clean.
#
# THIS DIRECTORY MUST EXIST BEFORE SUBMIT. Nothing in this script can create it in time:
#   mkdir -p /work/hdd/bbub/mjvolk3/slurm-logs/019-expr-v10
#SBATCH --output=/work/hdd/bbub/mjvolk3/slurm-logs/019-expr-v10/%x_%A_%a.out
#SBATCH --error=/work/hdd/bbub/mjvolk3/slurm-logs/019-expr-v10/%x_%A_%a.out

set -euo pipefail

PROJECT_ROOT="${V10_PROJECT_ROOT:-/projects/bbub/mjvolk3/torchcell}"
# PER-RUN LOGS SIT WITH THE SLURM LOGS, outside every checkout, for the same reason the
# --output directive does: a run's log is evidence about the run and must not be something a
# git operation on the source tree can delete. This defaults to the same directory --output
# names, so both halves of a task's output land together.
OUT_DIR="${V10_LOG_DIR:-/work/hdd/bbub/mjvolk3/slurm-logs/019-expr-v10}"
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
  # THE ROUND. 16 cells x 2 replicates = 32 runs over 8 array tasks, each a whole node
  # running four runs in ONE round, one run per GPU.
  #
  # WHY ONE ROUND AND NOT TWO, which is a change from 64 runs at 700 epochs. Two sequential
  # rounds per GPU is what forced the 700-epoch budget: 48 h of wall, minus a startup per
  # round, halved. Measuring the replicate spread at 700 (short_budget_spread.py, extended
  # down to 350) showed that trade is a bad one. The spread is 0.0166 at 700 against 0.0117
  # at 1,500, so halving the replicate count costs sqrt(2) in the standard error while the
  # longer budget returns almost exactly that same factor in sd. The smallest detectable
  # main effect is 0.0117 either way, and one round buys 2.3x the training budget for it.
  # 700 epochs is also the least stable point on the whole budget grid (sd 0.0058 at 500,
  # 0.0166 at 700, 0.0099 at 1,000), which is what a curve caught mid-rise at different
  # rates produces, and it leaves every cell at a mean of 0.1411 against an incumbent of
  # 0.2433. What is lost is per-cell resolution, which weakens from 16v16 to 8v8; main
  # effects are what this round exists to measure.
  grid)   CELLS=($(seq 0 15)); REPS=2; ROUNDS=1
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
# WHY PARITY HALVES ARE NO LONGER THE FIX. The even-parity half {0,3,5,6,9,10,12,15} is
# balanced four-and-four on every factor, so it blocked correctly when a node took EIGHT
# runs. At one round a node takes FOUR, which splits that half again, and the split is not
# innocent: the first four are {0,3,5,6}, every one of them weight decay 1e-8, and the next
# four are {9,10,12,15}, every one 1e-4. That is the original defect exactly, reintroduced
# by the smaller block. A slow or hot node would read out as a weight-decay effect.
#
# THE FIX AT BLOCK SIZE FOUR. Group by the two INTERACTION bits, g = (b0^b1) + 2*(b2^b3),
# which partitions the sixteen cells into four groups of four:
#   g=0 {0,3,12,15}   g=1 {1,2,13,14}   g=2 {4,7,8,11}   g=3 {5,6,9,10}
# Each group holds exactly two cells at each level of each of the four factors, so a
# node-level effect lands equally on both levels of every factor and cancels out of every
# main effect. Verified by the balance assertion below rather than trusted.
#
# WHAT IT COSTS, because blocking is never free. The block is defined BY the two-factor
# interactions embedding x trunk and readout x weight-decay, so those two interactions are
# confounded with node and cannot be estimated. Main effects, which is what this round
# exists to measure, are clean. Spending the interactions is the standard price and it is
# the right one here.

# GPU COUNT COMES FROM THE ALLOCATION, not a constant. Slurm sets SLURM_GPUS_ON_NODE to
# what this task actually got, so the same script is correct whether it is handed a whole
# 4-GPU node for the grid or a single card for a short canary. Hardcoding 4 meant a canary
# submitted with --gpus-per-node=1 would have tried to pin runs to cards 1-3 that it does
# not own, and every one of those would have failed on an invalid device.
GPUS_PER_NODE="${SLURM_GPUS_ON_NODE:-4}"
PER_TASK=$((GPUS_PER_NODE * ROUNDS))

if [[ "$STAGE" == "grid" ]]; then
  if [[ "$PER_TASK" -ne 4 ]]; then
    echo "FATAL: grid blocking assumes 4 runs per task, got PER_TASK=$PER_TASK" >&2
    echo "       (GPUS_PER_NODE=$GPUS_PER_NODE x ROUNDS=$ROUNDS)" >&2
    exit 1
  fi
  JOBS=()
  for rep in $(seq 0 $((REPS - 1))); do
    for g in 0 1 2 3; do
      for c in "${CELLS[@]}"; do
        gc=$(( (((c & 1) ^ ((c >> 1) & 1))) + 2 * ((((c >> 2) & 1) ^ ((c >> 3) & 1))) ))
        [[ "$gc" -eq "$g" ]] && JOBS+=("$c:$rep")
      done
    done
  done
  # BALANCE ASSERTION. Every block of PER_TASK consecutive jobs must be split two-and-two on
  # each of the four bits. Checked rather than asserted in a comment, because the failure is
  # silent: a confounded round still produces a full set of numbers and a clean-looking
  # leaderboard, and nothing downstream can recover the design after the fact.
  for (( b = 0; b < ${#JOBS[@]}; b += PER_TASK )); do
    for bit in 0 1 2 3; do
      ones=0
      for (( k = b; k < b + PER_TASK; k++ )); do
        cell="${JOBS[k]%%:*}"
        ones=$(( ones + (((cell >> bit) & 1)) ))
      done
      if [[ "$ones" -ne 2 ]]; then
        echo "FATAL: block at $b is $ones/4 on bit $bit, not 2/4; blocking is broken." >&2
        exit 1
      fi
    done
  done
else
  JOBS=()
  for rep in $(seq 0 $((REPS - 1))); do
    for c in "${CELLS[@]}"; do
      JOBS+=("$c:$rep")
    done
  done
fi

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

# THE JOB CARRIES ITS OWN ENVIRONMENT INSTEAD OF INHERITING A `.env`, and grid 21814726 is
# why. `.env` is gitignored, so it exists in the long-lived shared checkout as an untracked
# file and does NOT exist in a fresh clone. Every run of that grid loaded the dataset for
# half an hour and then died on `KeyError: 'EXPERIMENT_ROOT'`. Copying the file into the
# clone would work once and rot silently, and the shared copy is already wrong for this job
# in two ways: it pins DATA_ROOT to /work/hdd rather than the /scratch build this round
# uses, and it points EXPERIMENT_ROOT and ASSET_IMAGES_DIR back into the shared checkout,
# which is the very tree a branch switch was allowed to move out from under a running job.
# Deriving all of it from PROJECT_ROOT keeps the job self-contained. `load_dotenv` does not
# override variables that are already set, so a `.env` appearing later cannot silently
# change any of these.
#
# W&B runs ONLINE here (Delta compute has outbound network, unlike IGB) and authenticates
# from ~/.netrc in $HOME, which no clone or branch switch touches.
export EXPERIMENT_ROOT="$PROJECT_ROOT/experiments"
export ASSET_IMAGES_DIR="$PROJECT_ROOT/notes/assets/images"
export MPLSTYLE_PATH="$PROJECT_ROOT/torchcell/torchcell.mplstyle"
export WANDB_MODE=online
mkdir -p "$ASSET_IMAGES_DIR"
for v in EXPERIMENT_ROOT ASSET_IMAGES_DIR MPLSTYLE_PATH; do
  p="${!v}"
  if [ ! -e "$p" ]; then
    echo "FATAL: $v=$p does not exist." >&2
    exit 1
  fi
done
if ! grep -q "api.wandb.ai" "$HOME/.netrc" 2>/dev/null; then
  echo "FATAL: no api.wandb.ai entry in ~/.netrc and WANDB_MODE=online; run 'wandb login'." >&2
  exit 1
fi
# THE CONFIG MUST EXIST BEFORE ANY GPU IS TOUCHED. Hydra resolves --config-name inside each
# per-run python process, so a missing config surfaces as 64 separate late failures across 8
# nodes rather than as one refused submit. That is exactly how grid 21796128 died: the shared
# checkout was switched to another branch mid-flight and every run reported
# "Cannot find primary config 'cgt_expr_v10_grid'" while the tasks sat in their stagger loop
# looking alive. Checking the file here costs nothing and localizes the failure to the launch.
STAGE_CONFIG_FILE="$PROJECT_ROOT/experiments/019-simb-multimodal/conf/$STAGE_CONFIG.yaml"
if [ ! -f "$STAGE_CONFIG_FILE" ]; then
  echo "FATAL: $STAGE_CONFIG_FILE is missing." >&2
  echo "       PROJECT_ROOT is on branch $(git -C "$PROJECT_ROOT" rev-parse --abbrev-ref HEAD 2>&1)." >&2
  echo "       Point V10_PROJECT_ROOT at a checkout this job OWNS, never a shared one." >&2
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
# which resolves through a meta-path finder rather than through sys.path. Measured
# 2026-09-04 on the GilaHyper env: setuptools APPENDS `_EditableFinder` to `sys.meta_path`,
# so it sits at index 4 BEHIND `PathFinder` at index 3, and `PathFinder` is what reads
# PYTHONPATH. PYTHONPATH therefore wins, which is what makes PROJECT_ROOT effective. That
# was measured on GilaHyper's env and NOT on Delta's, so this asserts the outcome rather
# than assuming the mechanism carries over. A finder that pointed elsewhere would train a
# different checkout's source while the log still printed this checkout's git hash: an
# unattributable result that looks completely normal. The check costs one interpreter
# startup and turns that into an immediate failure before any GPU is spent.
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
