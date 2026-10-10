#!/bin/bash
# experiments/040-inhibitor-synergy-wetlab/scripts/gh_train_mixture.slurm
# [[experiments.040-inhibitor-synergy-wetlab.scripts.gh_train_mixture]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/040-inhibitor-synergy-wetlab/scripts/gh_train_mixture.slurm
#
# ONE SWEEP FILE OF train_mixture.py ON ONE GPU, ON GILAHYPER. PARALLEL processes share
# the card, each taking every PARALLEL-th config of the sweep (per-worker log
# <job>_<name>_w<k>.out); PARALLEL=1 runs the configs in sequence in this log. The
# pattern is experiment 038's gh_train_factorized.slurm.
#
#   sbatch --job-name=040-r1-van_ex21_film --export=ALL,SWEEP=conf/mixture/r1_mix_van_ex21_film_L1_lh1.yaml,PARALLEL=2 \
#     experiments/040-inhibitor-synergy-wetlab/scripts/gh_train_mixture.slurm
#
# SWEEP     path relative to the experiment folder
# ONLY      optional space-separated config names (PARALLEL=1 only)
# PARALLEL  processes per GPU, default 1
#SBATCH --job-name=040-mix
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/home/michaelvolk/Documents/projects/torchcell.worktrees/exp/040-inhibitor-synergy-wetlab/experiments/040-inhibitor-synergy-wetlab/slurm/output/%j_%x.out
#SBATCH --error=/home/michaelvolk/Documents/projects/torchcell.worktrees/exp/040-inhibitor-synergy-wetlab/experiments/040-inhibitor-synergy-wetlab/slurm/output/%j_%x.out
#SBATCH --no-requeue

set -euo pipefail
SWEEP=${SWEEP:?SWEEP is required}
ONLY=${ONLY:-}
PARALLEL=${PARALLEL:-1}
PROJECT_ROOT=/home/michaelvolk/Documents/projects/torchcell.worktrees/exp/040-inhibitor-synergy-wetlab
PYTHON=/home/michaelvolk/miniconda3/envs/torchcell/bin/python
EXP=$PROJECT_ROOT/experiments/040-inhibitor-synergy-wetlab
cd "$PROJECT_ROOT"
echo "Running on $(hostname) at $(date -Is), $(git rev-parse --short HEAD), SWEEP=$SWEEP ONLY=$ONLY PARALLEL=$PARALLEL"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
export PYTHONPATH="$PROJECT_ROOT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

# Every input, checked before a card is spent on it: the sweep, the model config, the
# cell table, the 031 fingerprints, the wet-lab wells, the model-free references, and the
# Vanacloig dev store the IC30 doses are read from.
for required in \
  "$EXP/$SWEEP" \
  "$EXP/conf/mixture/cgt_model.yaml" \
  "$EXP/results/wetlab_wells.csv" \
  "$EXP/results/mixture_scores.csv" \
  "$EXP/results/isobole_summary.csv" \
  "${TC040_CELL_TABLE:-/scratch/projects/torchcell-scratch/experiments/033-env-chemgen-pooled/cell_table_002/cell_table.parquet}"; do
  [[ -f "$required" ]] || { echo "ERROR: required file missing: $required" >&2; exit 1; }
done
for required_dir in \
  "${TC040_EMBEDDING_DIR:-/home/michaelvolk/Documents/projects/torchcell.worktrees/exp/031-env-chemgen-vanacloig-hillenmeyer/experiments/031-env-chemgen-inhibitor-tolerance/results/embeddings}" \
  "${TC040_VANACLOIG_STORE:-/scratch/projects/torchcell-scratch/data/torchcell/env_chemgen_vanacloig2022/processed}/interned"; do
  [[ -d "$required_dir" ]] || { echo "ERROR: required tree missing: $required_dir" >&2; exit 1; }
done
"$PYTHON" -c "import torchcell; p = torchcell.__file__; assert p.startswith('$PROJECT_ROOT'), p; print('torchcell from', p)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

if [ "$PARALLEL" -eq 1 ]; then
  if [ -n "$ONLY" ]; then
    # shellcheck disable=SC2086
    $PYTHON "$EXP/scripts/train_mixture.py" --sweep "$EXP/$SWEEP" --only $ONLY
  else
    $PYTHON "$EXP/scripts/train_mixture.py" --sweep "$EXP/$SWEEP"
  fi
else
  NAMES=$($PYTHON -c "import yaml,sys; print(' '.join(c['name'] for c in yaml.safe_load(open(sys.argv[1]))))" "$EXP/$SWEEP")
  read -r -a ALL <<< "$NAMES"
  for ((k = 0; k < PARALLEL; k++)); do
    SUBSET=()
    for ((i = k; i < ${#ALL[@]}; i += PARALLEL)); do SUBSET+=("${ALL[$i]}"); done
    if [ ${#SUBSET[@]} -eq 0 ]; then continue; fi
    echo "worker $k: ${SUBSET[*]}"
    $PYTHON "$EXP/scripts/train_mixture.py" --sweep "$EXP/$SWEEP" --only "${SUBSET[@]}" \
      > "$EXP/slurm/output/${SLURM_JOB_ID}_${SLURM_JOB_NAME}_w${k}.out" 2>&1 &
    # concurrent first readers of one parquet have died when started in the same second
    sleep 5
  done
  wait
  grep -hE "^  (ex2|isobole|vanacloig)" "$EXP/slurm/output/${SLURM_JOB_ID}_${SLURM_JOB_NAME}"_w*.out || true
fi
echo "finished at $(date -Is)"
