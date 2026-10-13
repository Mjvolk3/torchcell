#!/bin/bash
# experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm
# [[experiments.040-inhibitor-synergy-wetlab.scripts.delta_train_mixture]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm
#
# ONE SWEEP FILE OF train_mixture.py ON ONE A40, ON DELTA. The Delta twin of
# gh_train_mixture.slurm, written from experiment 038's delta_train_factorized.slurm.
# What differs is the machine: the code tree is a plain rsync of the 040 worktree on
# /scratch (no git), the data root is the Delta scratch tree, the interpreter is the
# /work/hdd env called by its absolute path, and the three inputs that live at GilaHyper
# absolute paths (the 033 cell table, the 031 embedding tables, the Vanacloig dev store)
# are pointed at their rsynced copies through TC040_CELL_TABLE, TC040_EMBEDDING_DIR and
# TC040_VANACLOIG_STORE.
#
#   SWEEP     path relative to the experiment folder (conf/mixture/<round>.yaml)
#   ONLY      optional space-separated config names (PARALLEL=1 only)
#   PARALLEL  processes per GPU, default 1
#
#   ssh delta
#   cd /scratch/bbub/mjvolk3/torchcell-040
#   mkdir -p experiments/040-inhibitor-synergy-wetlab/slurm/output
#   sbatch --job-name=040-r1-van_ex21_film --export=ALL,SWEEP=conf/mixture/r1_mix_van_ex21_film_L1_lh1.yaml,PARALLEL=2 \
#          experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm
#
# THE mkdir IS NOT OPTIONAL: slurm will not create the --output directory.
#
# RSYNC THE TREE FIRST. The 040 tree needs, beside the worktree itself: the 033 cell
# table at $DATA_ROOT/experiments/033-env-chemgen-pooled/cell_table_002/, the 031
# embedding tables into experiments/031-env-chemgen-inhibitor-tolerance/results/embeddings
# inside the tree (as the 038 tree carries them), the Vanacloig dev store's processed
# directory (lmdb + interned, about 310 KB of interned environments beside the records),
# and the genome, GO, STRING and TFLink roots under $DATA_ROOT/data.
#
# ACCOUNT: bfjt-delta-gpu, the GPU account with no running campaign on 2026-10-10
# (bgcg carries the 031/032 packs). Override with --account.
# PARTITION: gpuA40x4 (12 h limit). Smokes go to gpuA40x4-interactive (1 h).
# NEVER a -preempt partition: jobs there have sat for days.
#SBATCH --partition=gpuA40x4
#SBATCH --account=bfjt-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64g
#SBATCH --time=12:00:00
#SBATCH --job-name=040-mix
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/scratch/bbub/mjvolk3/torchcell-040/experiments/040-inhibitor-synergy-wetlab/slurm/output/%j_%x.out
#SBATCH --error=/scratch/bbub/mjvolk3/torchcell-040/experiments/040-inhibitor-synergy-wetlab/slurm/output/%j_%x.out

set -u

# Absolute, never SLURM_SUBMIT_DIR: `ssh delta 'sbatch ...'` submits from $HOME.
PROJECT_ROOT="/scratch/bbub/mjvolk3/torchcell-040"
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
export TC040_CELL_TABLE="$DATA_ROOT/experiments/033-env-chemgen-pooled/cell_table_002/cell_table.parquet"
export TC040_EMBEDDING_DIR="$PROJECT_ROOT/experiments/031-env-chemgen-inhibitor-tolerance/results/embeddings"
export TC040_VANACLOIG_STORE="$DATA_ROOT/data/torchcell/env_chemgen_vanacloig2022/processed"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS=ignore
export WANDB__SERVICE_WAIT=600
export WANDB_ENTITY="${WANDB_ENTITY:-zhao-group}"
export WANDB_MODE="${WANDB_MODE:-online}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

EXP="$EXPERIMENT_ROOT/040-inhibitor-synergy-wetlab"
SWEEP=${SWEEP:?SWEEP is required}
ONLY=${ONLY:-}
PARALLEL=${PARALLEL:-1}
LOG_DIR="$EXP/slurm/output"
mkdir -p "$LOG_DIR" "$DATA_ROOT/wandb-experiments/040-inhibitor-synergy-wetlab"

echo "=========================================================="
echo "host        : $(hostname)   job ${SLURM_JOB_ID:-none}   $(date -Is)"
echo "project     : $PROJECT_ROOT"
echo "DATA_ROOT   : $DATA_ROOT"
echo "python      : $DELTA_PY"
echo "sweep       : $SWEEP ONLY=$ONLY PARALLEL=$PARALLEL"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "=========================================================="
# Every check has cost a real job on this cluster: the interpreter, the genome database
# as a FILE, the graph roots, and the rsynced inputs.
[[ -x "$DELTA_PY" ]] || { echo "ERROR: interpreter not found: $DELTA_PY" >&2; exit 1; }
for required in \
  "$DATA_ROOT/data/sgd/genome/data.db" \
  "$DATA_ROOT/data/go/go.obo" \
  "$TC040_CELL_TABLE" \
  "$TC040_EMBEDDING_DIR/fcfp4_count.npz" \
  "$EXP/conf/mixture/cgt_model.yaml" \
  "$EXP/results/wetlab_wells.csv" \
  "$EXP/results/mixture_scores.csv" \
  "$EXP/results/isobole_summary.csv" \
  "$EXP/$SWEEP"; do
  [[ -f "$required" ]] || { echo "ERROR: required file missing: $required" >&2; exit 1; }
done
for required_dir in \
  "$DATA_ROOT/data/string" \
  "$DATA_ROOT/data/tflink" \
  "$TC040_VANACLOIG_STORE/interned" \
  "$TC040_VANACLOIG_STORE/lmdb"; do
  [[ -d "$required_dir" ]] || { echo "ERROR: required tree missing: $required_dir" >&2; exit 1; }
done
# The top-level package is not enough. This env carries an EDITABLE install of the stale
# checkout /projects/bbub/mjvolk3/torchcell, and its finder answers for any torchcell.X
# that PathFinder cannot resolve under PYTHONPATH. An rsync whose bare --exclude 'data'
# dropped torchcell/data therefore imported torchcell.data from that stale tree in
# silence (smoke 22787137). Check the submodules this trainer touches by file path.
"$DELTA_PY" -c "import sys, torchcell, torchcell.data, torchcell.datasets, torchcell.graph, torchcell.models, torchcell.sequence; mods = [torchcell, torchcell.data, torchcell.datasets, torchcell.graph, torchcell.models, torchcell.sequence]; bad = [m.__name__ for m in mods if not (m.__file__ or '').startswith('$PROJECT_ROOT')]; print('\n'.join(m.__name__ + ' from ' + str(m.__file__) for m in mods)); sys.exit('ERROR: resolved outside the tree: ' + ', '.join(bad)) if bad else None" || exit 1
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

if [ "$PARALLEL" -eq 1 ]; then
  if [ -n "$ONLY" ]; then
    # shellcheck disable=SC2086
    "$DELTA_PY" "$EXP/scripts/train_mixture.py" --sweep "$EXP/$SWEEP" --only $ONLY
  else
    "$DELTA_PY" "$EXP/scripts/train_mixture.py" --sweep "$EXP/$SWEEP"
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
    "$DELTA_PY" "$EXP/scripts/train_mixture.py" --sweep "$EXP/$SWEEP" --only "${SUBSET[@]}" \
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
