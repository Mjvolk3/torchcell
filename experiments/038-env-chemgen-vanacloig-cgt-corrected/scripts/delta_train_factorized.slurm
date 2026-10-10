#!/bin/bash
# experiments/038-env-chemgen-vanacloig-cgt-corrected/scripts/delta_train_factorized.slurm
# [[experiments.038-env-chemgen-vanacloig-cgt-corrected.scripts.delta_train_factorized]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/038-env-chemgen-vanacloig-cgt-corrected/scripts/delta_train_factorized.slurm
#
# ONE SWEEP FILE OF train_factorized.py ON ONE A40, ON DELTA. The Delta twin of
# gh_train_factorized.slurm: PARALLEL processes share the card, each taking every
# PARALLEL-th config of the sweep (per-worker log <job>_<name>_w<k>.out). What differs is
# the machine (see 031's delta_pack.slurm for the lessons): the code tree is a plain rsync
# of the 038 worktree on /scratch (no git), the data root is the Delta scratch tree, the
# interpreter is the /work/hdd env called by its absolute path, and the two inputs that
# live at GilaHyper absolute paths on the worktree (the 033 cell table, the 031 embedding
# tables) are pointed at their rsynced copies through TC038_CELL_TABLE and
# TC038_EMBEDDING_DIR.
#
#   SWEEP     path relative to the experiment folder (conf/factorized/<round>.yaml)
#   ONLY      optional space-separated config names (PARALLEL=1 only)
#   PARALLEL  processes per GPU, default 1
#
#   ssh delta
#   cd /scratch/bbub/mjvolk3/torchcell-038
#   mkdir -p experiments/038-env-chemgen-vanacloig-cgt-corrected/slurm/output
#   sbatch --job-name=038-r2w1-enc --export=ALL,SWEEP=conf/factorized/r10_envenc.yaml,PARALLEL=2 \
#          experiments/038-env-chemgen-vanacloig-cgt-corrected/scripts/delta_train_factorized.slurm
#
# THE mkdir IS NOT OPTIONAL: slurm will not create the --output directory.
#
# ACCOUNT: bfjt-delta-gpu, the GPU account with no running campaign on 2026-10-10
# (14,478 h of 20,947); bgcg carries the 031/032 packs. Override with --account.
# PARTITION: gpuA40x4 (12 h limit; the <= 12 h class waits a median 7.3 h). Smokes go to
# gpuA40x4-interactive (1 h). NEVER a -preempt partition.
# WALL CLOCK: a 15-config round of the 50-epoch recipe took 5 h 50 min at PARALLEL=2 on a
# GilaHyper RTX 6000 Ada (slurm 3080); an A40 is slower, so the full 12 h is requested.
#SBATCH --partition=gpuA40x4
#SBATCH --account=bfjt-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64g
#SBATCH --time=12:00:00
#SBATCH --job-name=038-fact
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/scratch/bbub/mjvolk3/torchcell-038/experiments/038-env-chemgen-vanacloig-cgt-corrected/slurm/output/%j_%x.out
#SBATCH --error=/scratch/bbub/mjvolk3/torchcell-038/experiments/038-env-chemgen-vanacloig-cgt-corrected/slurm/output/%j_%x.out

set -u

# Absolute, never SLURM_SUBMIT_DIR: `ssh delta 'sbatch ...'` submits from $HOME.
PROJECT_ROOT="/scratch/bbub/mjvolk3/torchcell-038"
DELTA_DATA_ROOT="${DELTA_DATA_ROOT:-/scratch/bbub/mjvolk3/torchcell}"
DELTA_PY="${DELTA_PY:-/work/hdd/bbub/miniconda3/envs/torchcell/bin/python}"
# The W&B settings of the Delta trees live in the 031 tree's env file (written at its
# setup with the Delta roots); the roots below are exported after it so they win.
DELTA_ENV_FILE="${DELTA_ENV_FILE:-/scratch/bbub/mjvolk3/torchcell-031/.env}"

cd "$PROJECT_ROOT" || exit 1
set -a; . "$DELTA_ENV_FILE"; set +a
export DATA_ROOT="$DELTA_DATA_ROOT"
export EXPERIMENT_ROOT="$PROJECT_ROOT/experiments"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export TC038_CELL_TABLE="$DATA_ROOT/experiments/033-env-chemgen-pooled/cell_table_002/cell_table.parquet"
export TC038_EMBEDDING_DIR="$PROJECT_ROOT/experiments/031-env-chemgen-inhibitor-tolerance/results/embeddings"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS=ignore
export WANDB__SERVICE_WAIT=600
export WANDB_ENTITY="${WANDB_ENTITY:-zhao-group}"
export WANDB_MODE="${WANDB_MODE:-online}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

EXP="$EXPERIMENT_ROOT/038-env-chemgen-vanacloig-cgt-corrected"
SWEEP=${SWEEP:?SWEEP is required}
ONLY=${ONLY:-}
PARALLEL=${PARALLEL:-1}
LOG_DIR="$EXP/slurm/output"
mkdir -p "$LOG_DIR" "$DATA_ROOT/wandb-experiments/038-env-chemgen-vanacloig-cgt-corrected"

echo "=========================================================="
echo "host        : $(hostname)   job ${SLURM_JOB_ID:-none}   $(date -Is)"
echo "project     : $PROJECT_ROOT"
echo "DATA_ROOT   : $DATA_ROOT"
echo "python      : $DELTA_PY"
echo "sweep       : $SWEEP ONLY=$ONLY PARALLEL=$PARALLEL"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "=========================================================="
# Every check has cost a real job on this cluster: the interpreter, the genome database as
# a FILE, the graph roots, and the two rsynced inputs.
[[ -x "$DELTA_PY" ]] || { echo "ERROR: interpreter not found: $DELTA_PY" >&2; exit 1; }
for required in \
  "$DATA_ROOT/data/sgd/genome/data.db" \
  "$DATA_ROOT/data/go/go.obo" \
  "$TC038_CELL_TABLE" \
  "$TC038_EMBEDDING_DIR/fcfp4_count.npz" \
  "$EXP/$SWEEP"; do
  [[ -f "$required" ]] || { echo "ERROR: required file missing: $required" >&2; exit 1; }
done
for required_dir in "$DATA_ROOT/data/string" "$DATA_ROOT/data/tflink"; do
  [[ -d "$required_dir" ]] || { echo "ERROR: required tree missing: $required_dir" >&2; exit 1; }
done
"$DELTA_PY" -c "import torchcell, sys; p = torchcell.__file__; assert p.startswith('$PROJECT_ROOT'), p; print('torchcell from', p)" || exit 1
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

if [ "$PARALLEL" -eq 1 ]; then
  if [ -n "$ONLY" ]; then
    # shellcheck disable=SC2086
    "$DELTA_PY" "$EXP/scripts/train_factorized.py" --sweep "$EXP/$SWEEP" --only $ONLY
  else
    "$DELTA_PY" "$EXP/scripts/train_factorized.py" --sweep "$EXP/$SWEEP"
  fi
else
  NAMES=$("$DELTA_PY" -c "import yaml,sys; print(' '.join(c['name'] for c in yaml.safe_load(open(sys.argv[1]))))" "$EXP/$SWEEP")
  read -r -a ALL <<< "$NAMES"
  pids=()
  for ((k = 0; k < PARALLEL; k++)); do
    SUBSET=()
    for ((i = k; i < ${#ALL[@]}; i += PARALLEL)); do SUBSET+=("${ALL[$i]}"); done
    [ ${#SUBSET[@]} -eq 0 ] && continue
    echo "worker $k: ${SUBSET[*]}"
    "$DELTA_PY" "$EXP/scripts/train_factorized.py" --sweep "$EXP/$SWEEP" --only "${SUBSET[@]}" \
      > "$LOG_DIR/${SLURM_JOB_ID}_${SLURM_JOB_NAME}_w${k}.out" 2>&1 &
    pids+=($!)
    # concurrent readers of one parquet/npz have died when started in the same second
    sleep 5
  done
  status=0
  for pid in "${pids[@]}"; do wait "$pid" || status=1; done
  echo "finished at $(date -Is) status $status"
  exit $status
fi
echo "finished at $(date -Is)"
