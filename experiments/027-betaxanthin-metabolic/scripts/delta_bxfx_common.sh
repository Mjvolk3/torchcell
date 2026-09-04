#!/bin/bash
# experiments/027-betaxanthin-metabolic/scripts/delta_bxfx_common.sh
# [[experiments.027-betaxanthin-metabolic]]
#
# The Delta environment + launch body shared by the 027 canary and the 027 grid array.
# Sourced, never executed directly.
#
# WHY 027 DOES NOT SOURCE `019-simb-multimodal/scripts/delta_grid_common.sh`
# --------------------------------------------------------------------------
# Two reasons, one of them a live bug:
#
#  1. THE SHARDING FIX IS MISSING THERE. The 019 include sets `OPTUNA_WORKER_ID` but points
#     every worker at ONE `OPTUNA_STORAGE` file and never exports `GRID_SHARD_COUNT`, so its
#     workers race optuna's WAITING-trial pop on SQLite. That race is measured, not
#     hypothetical: on the Delta grids `s03_L2_maskon_lr0.001_energy` ran FOUR times and
#     `bx_ctrl s01` twice, silently halving coverage. Only the IGB launcher was ever fixed.
#     This file exports `GRID_SHARD_COUNT` and gives every worker its own `_w<id>.db`.
#  2. It is wired to `metabolism_grid_runner.py`, which drives `train_cgt_multitask.py` --
#     the harness that cannot build a flux layer. 027 drives its own runner.
#
# Everything else below is copied from the 019 include because those values were validated
# interactively on Delta rather than guessed, and the comments say what each one cost.

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-/projects/bbub/mjvolk3/torchcell}"

# DATA_ROOT is the LARGE space, NOT /projects (tight quota) and NOT /work/hdd (which several
# older scripts in this repo still name). Confirmed on the Delta login node 2026-09-03: the
# tree carrying `data/torchcell/experiments/019-simb-multimodal/fig6_pigment_transfer` is
# /scratch/bbub/mjvolk3/torchcell. Getting this wrong is the single most expensive mistake
# available here, because `Neo4jCellDataset` skips the database only when `processed/`
# already exists -- point it at a root with no build and it tries to REBUILD, on a compute
# node with no Neo4j.
DELTA_DATA_ROOT="${DELTA_DATA_ROOT:-/scratch/bbub/mjvolk3/torchcell}"

# The env validated on Delta 2026-07-28, re-verified 2026-09-03: `torchcell` on /work/hdd,
# python3.13. NOT Delta's stock envs (python 3.11, SyntaxError on the repo's PEP 695
# generics). Called DIRECTLY -- no `conda activate` (that prefix has no bin/activate and the
# job silently falls back to system python3.9) and NO apptainer (the env's torch is built
# against the host CUDA).
DELTA_CONDA_BASE="${DELTA_CONDA_BASE:-/work/hdd/bbub/miniconda3}"
DELTA_CONDA_ENV="${DELTA_CONDA_ENV:-torchcell}"
DELTA_PY="$DELTA_CONDA_BASE/envs/$DELTA_CONDA_ENV/bin/python"

# DATALOADER WORKERS: 0 ON DELTA, and this is a measurement rather than a preference.
# The datamodule uses multiprocessing_context="spawn" (required -- CUDA is already
# initialized in the parent, so fork would corrupt the context), so every worker is a FRESH
# interpreter re-importing torch + PyG + torchcell from the parallel filesystem. On
# GilaHyper's local NVMe that costs seconds; on Delta the 2026-07-28 smoke test sat in
# `Sanity Checking` for 38 MINUTES without reaching a val batch. Same code, same dataset, 20x
# the startup cost -- the variable is the filesystem. At batch 128 the train split is ~29
# batches per epoch, so loading in the main process is a real option, not a crippled one.
export NUM_WORKERS="${NUM_WORKERS:-0}"

# 24 seeds. The arithmetic is in README.md: the measured paired SD of the arm contrast is
# 0.0554, so the 2-sigma minimum detectable effect is 0.090 at n=3 and 0.031 at n=24, against
# an excess-over-null of +0.043 that 026 could not resolve.
export GRID_ROUNDS="${GRID_ROUNDS:-24}"
export GRID_SEED_BASE="${GRID_SEED_BASE:-2700}"
# The floor a worker assumes for the FIRST cell, before it has measured one. 200 epochs at
# the GilaHyper-measured 51.4 s/epoch with the 1.44x two-per-GPU penalty is 4.1 h, so 4.5 h is
# that plus a margin. After cell 1 the guard uses the LONGEST cell actually observed, which is
# what matters if Delta runs slower.
export GRID_MIN_TRIAL_S="${GRID_MIN_TRIAL_S:-16200}"
# Reserved after training for the test pass + per-gene dump + wandb teardown. The test pass
# is 933 records and the dump is one JSON write, so this is an order of magnitude of headroom
# -- but it must not be zero, or the wall clock SIGTERMs the process during the dump that is
# the deliverable.
export GRID_TEARDOWN_S="${GRID_TEARDOWN_S:-600}"
# 2 runs per GPU. 019 measured 2/GPU at 1.44x per-run slowdown for 1.39x aggregate
# throughput, so packing is a net win, and it also means every worker starts at t=0 and
# shares the full window -- a wall-clock kill then leaves many partial curves rather than
# half the queue never started. The model is hidden=32 / L=2, far smaller than the L=6
# settings that made 2/GPU a memory question on the 020 grid.
export GRID_WORKERS_PER_GPU="${GRID_WORKERS_PER_GPU:-2}"
export DELTA_GPUS_PER_NODE="${DELTA_GPUS_PER_NODE:-4}"

bxfx_env() {
  export DATA_ROOT="$DELTA_DATA_ROOT"
  export EXPERIMENT_ROOT="$PROJECT_ROOT/experiments"
  export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export PYTHONUNBUFFERED=1
  export PYTHONWARNINGS=ignore
  # Delta compute nodes have internet, so W&B is ONLINE. No offline/sync dance.
  export WANDB__SERVICE_WAIT=600
  export WANDB_PROJECT="${WANDB_PROJECT:-torchcell_027_bxfx}"
  # Set explicitly. Unset, wandb falls back to whatever default entity the Delta env
  # carries, and 026's runs landed under zhao-group; a split between two entities makes
  # the arms unjoinable at analysis time.
  export WANDB_ENTITY="${WANDB_ENTITY:-zhao-group}"
  # Delta compute nodes DO reach the internet: the 2026-07-22 Optuna launch and the 020_v4
  # grid both produced live run lists. An older comment in delta_pigment_transfer.slurm
  # claims the opposite; it is stale, and WANDB__SERVICE_WAIT above bounds the damage if it
  # ever becomes true again.
  export WANDB_MODE="${WANDB_MODE:-online}"
  # 8 processes at num_workers=0 on a 32-CPU allocation of a 64-core node each default to
  # the node's core count, which oversubscribes by ~16x.
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
  export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

  EXP_DIR="$PROJECT_ROOT/experiments/027-betaxanthin-metabolic"
  OPTUNA_DIR="$EXP_DIR/optuna"
  RUNNER="$EXP_DIR/scripts/grid_runner.py"
  TRAINER="$EXP_DIR/scripts/train_bx.py"
  DATASET_DIR="$DATA_ROOT/data/torchcell/experiments/019-simb-multimodal/fig6_pigment_transfer"
  OED_DIR="$DATA_ROOT/data/enzyme_kinetics/open_enzyme_database/scerevisiae"
  # slurm does not create the --output directory, and a missing one kills the job at t=0 with
  # NO LOG ANYWHERE. This is a backstop; the .slurm headers name the mkdir to run by hand.
  mkdir -p "$OPTUNA_DIR" "$EXP_DIR/slurm/output" "$EXP_DIR/results"
}

bxfx_preflight() {
  echo "=========================================================="
  echo "host        : $(hostname)"
  echo "job         : ${SLURM_JOB_ID:-none}  array=${SLURM_ARRAY_TASK_ID:-none}"
  echo "project     : $PROJECT_ROOT"
  echo "DATA_ROOT   : $DATA_ROOT"
  echo "python      : $DELTA_PY"
  echo "rounds      : $GRID_ROUNDS   seed_base=$GRID_SEED_BASE   num_workers=$NUM_WORKERS"
  echo "=========================================================="

  # EVERY CHECK BELOW HAS COST A REAL JOB, ON THIS CLUSTER OR THE LAST ONE.
  [[ -x "$DELTA_PY" ]] || {
    echo "ERROR: interpreter not found: $DELTA_PY" >&2; exit 1; }
  # The genome's gffutils SQLite, checked as a FILE. The directory existing is not enough
  # and this is not hypothetical: job 21797666 passed a `-d` check on data/sgd/genome and
  # then lost every worker on three of nine array tasks to
  # `FileNotFoundError: .../data/sgd/genome/data.db`, because the directory shipped without
  # the database. A worker that hits this dies AFTER the whole import, so the cost is one
  # full Lustre import per worker before anything reports.
  for required in \
    "$DATA_ROOT/data/sgd/genome/data.db" \
    "$DATA_ROOT/data/go/go.obo"; do
    [[ -f "$required" ]] || {
      echo "ERROR: required file missing: $required" >&2
      echo "       rsync it from the build host before resubmitting." >&2
      exit 1
    }
  done
  for required_dir in \
    "$DATA_ROOT/data/scerevisiae/protT5_embedding" \
    "$DATA_ROOT/data/string" \
    "$DATA_ROOT/data/tflink"; do
    [[ -d "$required_dir" ]] || {
      echo "ERROR: required tree missing: $required_dir" >&2
      exit 1
    }
  done

  [[ -d "$DATASET_DIR" ]] || {
    echo "ERROR: dataset tree missing: $DATASET_DIR" >&2
    echo "       DATA_ROOT in force is '$DATA_ROOT'. Resubmit with DELTA_DATA_ROOT=<root>;" >&2
    echo "       do NOT edit the launcher." >&2
    exit 1; }
  [[ -f "$RUNNER" && -f "$TRAINER" ]] || {
    echo "ERROR: 027 scripts missing under $EXP_DIR/scripts" >&2; exit 1; }

  # THE 027-SPECIFIC ONE, and the reason it is a hard gate rather than a warning.
  # `resolve_kcat_table` CATCHES FileNotFoundError on a missing Open Enzyme Database mirror
  # and returns an empty record list, so every kcat silently becomes the organism default:
  # the "enzyme-constrained" arm runs with no enzyme constraints, completes normally, and
  # reports a plausible number. The mirror is 512 KB and was NOT on Delta as of 2026-09-03.
  [[ -f "$OED_DIR/oed_records.json" ]] || {
    echo "ERROR: Open Enzyme Database mirror missing: $OED_DIR" >&2
    echo "       Every flux arm would run with default kcat and still report a number." >&2
    echo "       rsync it from GilaHyper (512 KB) -- see README.md, Prerequisites." >&2
    exit 1; }
  [[ -f "$PROJECT_ROOT/torchcell/metabolism/flux_layer.py" ]] || {
    echo "ERROR: torchcell/metabolism/flux_layer.py absent -- this checkout predates the" >&2
    echo "       026 landing. git fetch && checkout the branch carrying 027." >&2
    exit 1; }
  nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
  echo "=========================================================="
}

# Launch this NODE's share of a sharded grid.
#   $1 global shard count   $2 this node's first worker id   $3 job deadline (unix epoch)
bxfx_run_node() {
  local shard_count="$1" first_worker="$2" deadline="$3"
  export GRID_SHARD_COUNT="$shard_count"
  export GRID_DEADLINE_EPOCH="$deadline"
  local n_local=$((DELTA_GPUS_PER_NODE * GRID_WORKERS_PER_GPU))

  echo "node workers: $n_local (ids $first_worker..$((first_worker + n_local - 1))) of $shard_count"

  # Build every local worker's queue in ONE process, before any GPU work.
  #
  # This used to be a loop of eight `--create-only` calls, on the reasoning that a process
  # "spends >15 s importing torch". MEASURED ON DELTA that import is roughly 50 MINUTES off
  # Lustre, so the loop spent about 6.7 hours of a 20-hour allocation writing empty SQLite
  # files and job 21797666 reached its first training step never. The timestamps are in its
  # log: w64 04:03, w65 04:55, w66 05:45, w67 06:34, w68 07:16, w69 07:55.
  #
  # One process pays the import once. The DDL is still not raced: each study is a separate
  # file, written sequentially inside the loop. This remains the import gate, so a shard
  # count that does not divide the arm count still exits before any GPU work.
  OPTUNA_WORKER_ID=$first_worker \
    OPTUNA_STORAGE="sqlite:///$OPTUNA_DIR/optuna_bxfx_w${first_worker}.db" \
    "$DELTA_PY" "$RUNNER" --create-batch "$first_worker" "$n_local" \
    "sqlite:///$OPTUNA_DIR/optuna_bxfx_w{worker}.db" || exit 1
  local w

  local i=0
  for ((w = first_worker; w < first_worker + n_local; w++)); do
    local dev=$((i % DELTA_GPUS_PER_NODE))
    CUDA_VISIBLE_DEVICES=$dev \
      OPTUNA_WORKER_ID=$w \
      OPTUNA_STORAGE="sqlite:///$OPTUNA_DIR/optuna_bxfx_w${w}.db" \
      "$DELTA_PY" -u "$RUNNER" &
    i=$((i + 1))
    # Stagger: eight processes opening the same LMDB at once is a needless contention spike
    # at t=0 on a parallel filesystem.
    sleep 20
  done
  wait
}
