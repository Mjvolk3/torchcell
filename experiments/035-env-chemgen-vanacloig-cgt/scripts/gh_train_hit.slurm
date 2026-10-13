#!/bin/bash
#SBATCH --job-name=035-hit
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/home/michaelvolk/Documents/projects/torchcell.worktrees/exp/035-env-chemgen-vanacloig-cgt/experiments/035-env-chemgen-vanacloig-cgt/slurm/output/%j_%x.out
#SBATCH --error=/home/michaelvolk/Documents/projects/torchcell.worktrees/exp/035-env-chemgen-vanacloig-cgt/experiments/035-env-chemgen-vanacloig-cgt/slurm/output/%j_%x.out
#SBATCH --no-requeue

# One sweep file of train_hit.py on one GPU. The models are small (about 2 GB, the GPU
# idle between steps), so PARALLEL processes share the card, each taking every
# PARALLEL-th config of the sweep.
#
#   sbatch --job-name=035-r6-mix --export=ALL,SWEEP=conf/factorized/r6_mix.yaml,PARALLEL=4 \
#     experiments/035-env-chemgen-vanacloig-cgt/scripts/gh_train_hit.slurm
set -euo pipefail
SWEEP=${SWEEP:?SWEEP is required}
PARALLEL=${PARALLEL:-1}
PROJECT_ROOT=/home/michaelvolk/Documents/projects/torchcell.worktrees/exp/035-env-chemgen-vanacloig-cgt
PYTHON=/home/michaelvolk/miniconda3/envs/torchcell/bin/python
EXP=$PROJECT_ROOT/experiments/035-env-chemgen-vanacloig-cgt
cd "$PROJECT_ROOT"
echo "Running on $(hostname) at $(date -Is), $(git rev-parse --short HEAD), SWEEP=$SWEEP PARALLEL=$PARALLEL"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
export PYTHONPATH="$PROJECT_ROOT"
NAMES=$($PYTHON -c "import yaml,sys; print(' '.join(c['name'] for c in yaml.safe_load(open(sys.argv[1]))))" "$EXP/$SWEEP")
read -r -a ALL <<< "$NAMES"
for ((k = 0; k < PARALLEL; k++)); do
  SUBSET=()
  for ((i = k; i < ${#ALL[@]}; i += PARALLEL)); do SUBSET+=("${ALL[$i]}"); done
  if [ ${#SUBSET[@]} -eq 0 ]; then continue; fi
  echo "worker $k: ${SUBSET[*]}"
  $PYTHON "$EXP/scripts/train_hit.py" --sweep "$EXP/$SWEEP" --only "${SUBSET[@]}" \
    > "$EXP/slurm/output/${SLURM_JOB_ID}_${SLURM_JOB_NAME}_w${k}.out" 2>&1 &
done
wait
grep -hE ": \{" "$EXP/slurm/output/${SLURM_JOB_ID}_${SLURM_JOB_NAME}"_w*.out || true
echo "finished at $(date -Is)"
