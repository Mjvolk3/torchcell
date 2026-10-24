#!/bin/bash
# experiments/043-expression-proteome-morphology/scripts/alcf/aurora_env_043.sh
# [[experiments.043-expression-proteome-morphology.scripts.alcf.aurora_env_043]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/043-expression-proteome-morphology/scripts/alcf/aurora_env_043.sh
#
# Sourced by the Aurora PBS scripts of 043: interpreter, proxy, FLAT tile hierarchy, data
# root, and the git shim (the synced tree has no .git; `git rev-parse --short HEAD` reads
# .synced_commit, written by transfer_043.sh code). Defines `train_on_tile TILE LOG ARGS...`,
# which pins one tile with ZE_AFFINITY_MASK and runs train_043.py in the background.
TC_PREFIX=/flare/GenesisHackathonOct26/mjvolk3
TC_TREE=$TC_PREFIX/torchcell
PY=$TC_PREFIX/miniforge3/envs/torchcell/bin/python
TRAIN=$TC_TREE/experiments/043-expression-proteome-morphology/scripts/train_043.py

export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128
export HTTP_PROXY=$http_proxy HTTPS_PROXY=$https_proxy
export no_proxy=localhost,127.0.0.1,.alcf.anl.gov
export ZE_FLAT_DEVICE_HIERARCHY=FLAT

cd "$TC_TREE"
set -a
# shellcheck disable=SC1091
source .env
set +a
export DATA_ROOT=$TC_PREFIX/torchcell-data/043
export EXPERIMENT_ROOT=$TC_TREE/experiments
export PYTHONUNBUFFERED=1 PYTHONPATH="$TC_TREE"
export WANDB_MODE=online

SHIM=$(mktemp -d)
REAL_GIT=$(command -v git)
SYNCED=$(cut -c1-9 "$TC_TREE/.synced_commit")
cat > "$SHIM/git" <<EOF
#!/bin/bash
if [ "\$*" = "rev-parse --short HEAD" ]; then echo "$SYNCED"; exit 0; fi
exec "$REAL_GIT" "\$@"
EOF
chmod +x "$SHIM/git"
export PATH="$SHIM:$PATH"

train_on_tile() {
  local tile=$1 log=$2
  shift 2
  echo "tile $tile -> $log : $*"
  ZE_AFFINITY_MASK=$tile "$PY" "$TRAIN" device=xpu "$@" > "$log" 2>&1 &
}
