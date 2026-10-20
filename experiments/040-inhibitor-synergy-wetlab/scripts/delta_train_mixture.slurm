#!/bin/bash
# experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm
# [[experiments.040-inhibitor-synergy-wetlab.scripts.delta_train_mixture]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm
#
# ONE SWEEP FILE OF train_mixture.py ON ONE A40, ON DELTA. The Delta twin of
# gh_train_mixture.slurm, written from experiment 038's delta_train_factorized.slurm.
# What differs is the machine: the code tree is a plain rsync of the 040 worktree on
# /scratch (no git), the data root is the Delta scratch tree, the interpreter is the
# /work/hdd env called by its absolute path, and the ONE data directory the trainer reads
# beside the standard roots, the 040 store (build_040_store.py: the KG 4.0 query's
# processed LMDB and its manifest), is pointed at its rsynced copy through TC040_STORE.
# The store's bytes are checked against the manifest's sha256 before training.
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
# RSYNC THE TREE FIRST. The 040 tree needs, beside the worktree itself: the 040 store's
# processed directory (processed/lmdb/data.mdb and processed/store_manifest.json) at
# $DATA_ROOT/data/torchcell/experiments/040-inhibitor-synergy-wetlab/mixture_040/, and
# the genome, GO, STRING and TFLink roots under $DATA_ROOT/data. The manifest's sha256
# is what the preflight below and the trainer check the copy against.
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
# FROM ROUND 11 (2026-10-17) the data root and the interpreter are 040's own. Lustre
# target 106 (dltawork-OST006a) stopped answering on 2026-10-16, which left about one
# file in a hundred unreadable in the shared env (5,863 files), the shared data root
# (1,026) and the old tree (213); any job that touches one hangs to its time limit. The
# env was rebuilt package for package (33 conda builds, 263 pip pins) into
# envs/torchcell-r11, and the inputs were copied fresh from GilaHyper into
# torchcell-040-data. The shared paths are untouched for the other Delta trees.
DELTA_DATA_ROOT="${DELTA_DATA_ROOT:-/scratch/bbub/mjvolk3/torchcell-040-data}"
DELTA_PY="${DELTA_PY:-/scratch/bbub/mjvolk3/envs/torchcell-r11/bin/python}"
# The W&B settings of the Delta trees live in the 031 tree's env file (written at its
# setup with the Delta roots); the roots below are exported after it so they win.
DELTA_ENV_FILE="${DELTA_ENV_FILE:-/scratch/bbub/mjvolk3/torchcell-031/.env}"

cd "$PROJECT_ROOT" || exit 1
set -a; . "$DELTA_ENV_FILE"; set +a
export DATA_ROOT="$DELTA_DATA_ROOT"
export EXPERIMENT_ROOT="$PROJECT_ROOT/experiments"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export TC040_STORE="${TC040_STORE:-$DATA_ROOT/data/torchcell/experiments/040-inhibitor-synergy-wetlab/mixture_040}"
export TC040_WORKERS="${TC040_WORKERS:-${SLURM_CPUS_PER_TASK:-8}}"
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
echo "store       : $TC040_STORE"
echo "sweep       : $SWEEP ONLY=$ONLY PARALLEL=$PARALLEL"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "=========================================================="
# Every check has cost a real job on this cluster: the interpreter, the genome database
# as a FILE, the graph roots, and the rsynced inputs.
[[ -x "$DELTA_PY" ]] || { echo "ERROR: interpreter not found: $DELTA_PY" >&2; exit 1; }
for required in \
  "$DATA_ROOT/data/sgd/genome/data.db" \
  "$DATA_ROOT/data/go/go.obo" \
  "$TC040_STORE/processed/store_manifest.json" \
  "$TC040_STORE/processed/lmdb/data.mdb" \
  "$EXP/conf/mixture/cgt_model.yaml" \
  "$EXP/results/wetlab_wells.csv" \
  "$EXP/results/mixture_scores.csv" \
  "$EXP/results/isobole_summary.csv" \
  "$EXP/$SWEEP"; do
  [[ -f "$required" ]] || { echo "ERROR: required file missing: $required" >&2; exit 1; }
done
for required_dir in \
  "$DATA_ROOT/data/string" \
  "$DATA_ROOT/data/tflink"; do
  [[ -d "$required_dir" ]] || { echo "ERROR: required tree missing: $required_dir" >&2; exit 1; }
done
# The store's bytes against its manifest (size, then sha256). A copy that Lustre damaged
# fails here, in seconds of CPU, instead of hanging a card to the time limit.
"$DELTA_PY" -c "import sys; sys.path.insert(0, '$EXP/scripts'); from store_manifest_040 import verify_store; m = verify_store('$TC040_STORE'); print('store verified:', m.kg_release, m.n_processed_entries, 'records', m.data_mdb_sha256[:12])" || exit 1
# The top-level package is not enough. This env carries an EDITABLE install of the stale
# checkout /projects/bbub/mjvolk3/torchcell, and its finder answers for any torchcell.X
# that PathFinder cannot resolve under PYTHONPATH. An rsync whose bare --exclude 'data'
# dropped torchcell/data therefore imported torchcell.data from that stale tree in
# silence (smoke 22787137). Check the submodules this trainer touches by file path.
"$DELTA_PY" -c "import sys, torchcell, torchcell.data, torchcell.datasets, torchcell.graph, torchcell.models, torchcell.sequence; mods = [torchcell, torchcell.data, torchcell.datasets, torchcell.graph, torchcell.models, torchcell.sequence]; bad = [m.__name__ for m in mods if not (m.__file__ or '').startswith('$PROJECT_ROOT')]; print('\n'.join(m.__name__ + ' from ' + str(m.__file__) for m in mods)); sys.exit('ERROR: resolved outside the tree: ' + ', '.join(bad)) if bad else None" || exit 1
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

if [ "$PARALLEL" -eq 1 ]; then
  # THE JOB'S STATE IS THE TRAINER'S. Until round 11 this branch ended on the final echo,
  # so a trainer that died on a Lustre I/O error still left the job COMPLETED (13 of
  # round 11's jobs, 2026-10-17); the exit status is now carried out of the script.
  if [ -n "$ONLY" ]; then
    # shellcheck disable=SC2086
    "$DELTA_PY" "$EXP/scripts/train_mixture.py" --sweep "$EXP/$SWEEP" --only $ONLY
  else
    "$DELTA_PY" "$EXP/scripts/train_mixture.py" --sweep "$EXP/$SWEEP"
  fi
  status=$?
  echo "finished at $(date -Is) status $status"
  exit $status
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
