#!/bin/bash
#SBATCH --job-name=035-fact
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/home/michaelvolk/Documents/projects/torchcell.worktrees/exp/035-env-chemgen-vanacloig-cgt/experiments/035-env-chemgen-vanacloig-cgt/slurm/output/%j_%x.out
#SBATCH --error=/home/michaelvolk/Documents/projects/torchcell.worktrees/exp/035-env-chemgen-vanacloig-cgt/experiments/035-env-chemgen-vanacloig-cgt/slurm/output/%j_%x.out
#SBATCH --no-requeue

# One sweep file of train_factorized.py on one GPU. PARALLEL processes share the card,
# each taking every PARALLEL-th config of the sweep (per-worker output in
# <job>_<name>_w<k>.out); PARALLEL=1 runs the configs in sequence in this log.
#
#   sbatch --job-name=035-r8w1-sat --export=ALL,SWEEP=conf/factorized/r8_saturate_a.yaml,PARALLEL=3 \
#     experiments/035-env-chemgen-vanacloig-cgt/scripts/gh_train_factorized.slurm
#
# SWEEP     path relative to the experiment folder
# ONLY      optional space-separated config names (PARALLEL=1 only)
# PARALLEL  processes per GPU, default 1
set -euo pipefail
SWEEP=${SWEEP:?SWEEP is required}
ONLY=${ONLY:-}
PARALLEL=${PARALLEL:-1}
PROJECT_ROOT=/home/michaelvolk/Documents/projects/torchcell.worktrees/exp/035-env-chemgen-vanacloig-cgt
PYTHON=/home/michaelvolk/miniconda3/envs/torchcell/bin/python
EXP=$PROJECT_ROOT/experiments/035-env-chemgen-vanacloig-cgt
cd "$PROJECT_ROOT"
echo "Running on $(hostname) at $(date -Is), $(git rev-parse --short HEAD), SWEEP=$SWEEP ONLY=$ONLY PARALLEL=$PARALLEL"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
export PYTHONPATH="$PROJECT_ROOT"
if [ "$PARALLEL" -eq 1 ]; then
  if [ -n "$ONLY" ]; then
    # shellcheck disable=SC2086
    $PYTHON "$EXP/scripts/train_factorized.py" --sweep "$EXP/$SWEEP" --only $ONLY
  else
    $PYTHON "$EXP/scripts/train_factorized.py" --sweep "$EXP/$SWEEP"
  fi
else
  NAMES=$($PYTHON -c "import yaml,sys; print(' '.join(c['name'] for c in yaml.safe_load(open(sys.argv[1]))))" "$EXP/$SWEEP")
  read -r -a ALL <<< "$NAMES"
  for ((k = 0; k < PARALLEL; k++)); do
    SUBSET=()
    for ((i = k; i < ${#ALL[@]}; i += PARALLEL)); do SUBSET+=("${ALL[$i]}"); done
    if [ ${#SUBSET[@]} -eq 0 ]; then continue; fi
    echo "worker $k: ${SUBSET[*]}"
    $PYTHON "$EXP/scripts/train_factorized.py" --sweep "$EXP/$SWEEP" --only "${SUBSET[@]}" \
      > "$EXP/slurm/output/${SLURM_JOB_ID}_${SLURM_JOB_NAME}_w${k}.out" 2>&1 &
  done
  wait
  grep -hE ": \{" "$EXP/slurm/output/${SLURM_JOB_ID}_${SLURM_JOB_NAME}"_w*.out || true
fi
echo "finished at $(date -Is)"
