#!/bin/bash
# experiments/042-hpc-smoke-tests/scripts/write_remote_env.sh
# [[experiments.042-hpc-smoke-tests.scripts.write_remote_env]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/042-hpc-smoke-tests/scripts/write_remote_env
#
# Write a machine's `.env` into its TC_TREE (sync_code.sh excludes .env on purpose: the
# paths differ per machine and the local file carries API keys that do not travel).
#
#   bash experiments/042-hpc-smoke-tests/scripts/write_remote_env.sh polaris
#
# Only the keys a training job reads. W&B authenticates through ~/.netrc on the remote
# machine, so no key is written here.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/machines.sh" "${1:-}"

tc_ssh "mkdir -p '$TC_DATA_ROOT/data/torchcell' '$TC_TREE/notes/assets/images' && cat > '$TC_TREE/.env'" <<EOF
# torchcell .env for $TC_MACHINE, written by experiments/042-hpc-smoke-tests/scripts/write_remote_env.sh
WORKSPACE_DIR=$TC_TREE
PYTHON_PKG_REL_PATH=torchcell
PYTHON_PKG_TEST_REL_PATH=tests
DATA_ROOT=$TC_DATA_ROOT
ASSET_IMAGES_DIR=$TC_TREE/notes/assets/images
EXPERIMENT_ROOT=$TC_TREE/experiments
WANDB_MODE=online
TORCHCELL_DEBUG_TIMING=false
EOF
echo "wrote $TC_MACHINE:$TC_TREE/.env (DATA_ROOT=$TC_DATA_ROOT)"
