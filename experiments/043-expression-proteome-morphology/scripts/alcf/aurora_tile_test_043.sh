#!/bin/bash
# experiments/043-expression-proteome-morphology/scripts/alcf/aurora_tile_test_043.sh
# [[experiments.043-expression-proteome-morphology.scripts.alcf.aurora_tile_test_043]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/043-expression-proteome-morphology/scripts/alcf/aurora_tile_test_043.sh
#
# Short throwaway trainer run on one tile of a running grid node, W&B disabled, to tell a
# lodo-specific crash from an environment one (the relaunched lodo cells segfault right
# after the DataLoader fork, exit 139). Run ON the compute node:
#
#   bash aurora_tile_test_043.sh <tile> <log> <hydra overrides ...>
set -uo pipefail
tile=$1 log=$2
shift 2
HERE=/flare/GenesisHackathonOct26/mjvolk3/torchcell/experiments/043-expression-proteome-morphology/scripts/alcf
# shellcheck disable=SC1091
source "$HERE/aurora_env_043.sh"
export WANDB_MODE=disabled
ZE_AFFINITY_MASK=$tile "$PY" "$TRAIN" device=xpu data.store_version=v2 data.prep_version=prep_v4 \
  data.num_workers=4 data.sampler=records data.mark_cap=null budget.time_budget_hours=null \
  budget.max_epochs=1 "++budget.max_steps_per_epoch=40" data.batch_size=2 \
  ++optim.accumulate_steps=4 wandb.mode=disabled \
  paths.checkpoint_root=/flare/GenesisHackathonOct26/mjvolk3/torchcell-data/043/tile_test "$@" \
  > "$log" 2>&1
echo "exit $?" >> "$log"
