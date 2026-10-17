#!/bin/bash
# experiments/042-hpc-smoke-tests/scripts/install_env.sh
# [[experiments.042-hpc-smoke-tests.scripts.install_env]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/042-hpc-smoke-tests/scripts/install_env
#
# Runs ON a remote login node. Builds the torchcell environment the way Delta and IGB
# have it: our own Miniforge in project space, python 3.13, the GilaHyper torch stack
# (torch 2.11.0, torch_geometric >= 2.8, torch-scatter from the PyG wheel index), then
# `pip install -e` of the rsynced tree. Idempotent: every step checks before it acts.
#
#   TC_PREFIX=/eagle/GenesisHackathonOct26/mjvolk3 TC_ACCEL=cu128 \
#     bash /eagle/GenesisHackathonOct26/mjvolk3/torchcell/experiments/042-hpc-smoke-tests/scripts/install_env.sh
#
# TC_ACCEL=cu128  NVIDIA (Polaris, Perlmutter): torch from the cu128 index + torch-scatter.
# TC_ACCEL=xpu    Intel Max (Aurora): torch from the xpu index; torch-scatter has no XPU
#                 build, so PyG runs on its native scatter path there.
set -euo pipefail

: "${TC_PREFIX:?set TC_PREFIX to the project-space dir}"
: "${TC_ACCEL:?set TC_ACCEL to cu128 or xpu}"
TC_TREE="${TC_TREE:-$TC_PREFIX/torchcell}"
TORCH_VERSION="${TORCH_VERSION:-2.11.0}"
PY_VERSION="${PY_VERSION:-3.13}"
MF="$TC_PREFIX/miniforge3"
ENV_PREFIX="$MF/envs/torchcell"

# Keep every cache off the (small) home filesystem.
export PIP_CACHE_DIR="$TC_PREFIX/.cache/pip"
export CONDA_PKGS_DIRS="$TC_PREFIX/.cache/conda-pkgs"
export TMPDIR="$TC_PREFIX/.cache/tmp"
mkdir -p "$PIP_CACHE_DIR" "$CONDA_PKGS_DIRS" "$TMPDIR"

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

if [[ ! -x "$MF/bin/conda" ]]; then
  log "installing Miniforge3 -> $MF"
  curl -fsSL -o "$TMPDIR/miniforge.sh" \
    https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
  bash "$TMPDIR/miniforge.sh" -b -p "$MF"
fi
# shellcheck disable=SC1091
source "$MF/bin/activate"

if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
  log "creating env torchcell (python $PY_VERSION)"
  conda create -y -q -n torchcell "python=$PY_VERSION"
fi
PY="$ENV_PREFIX/bin/python"
log "python: $($PY --version)"

if ! "$PY" -c "import torch, sys; sys.exit(0 if torch.__version__.startswith('$TORCH_VERSION') else 1)" 2>/dev/null; then
  log "installing torch $TORCH_VERSION ($TC_ACCEL)"
  nice -n 10 "$PY" -m pip install -q "torch==$TORCH_VERSION" \
    --index-url "https://download.pytorch.org/whl/$TC_ACCEL"
fi
log "torch: $("$PY" -c 'import torch; print(torch.__version__)')"

# env/requirements_experiments.txt is RAPIDS (cudf, cuml): nothing in torchcell imports it,
# and its placeholder wheels run nvidia-smi at install time, which a login node has none of.
log "installing env/requirements.txt"
nice -n 10 "$PY" -m pip install -q -r "$TC_TREE/env/requirements.txt"

if [[ "$TC_ACCEL" == cu128 ]]; then
  if ! "$PY" -c "import torch_scatter" 2>/dev/null; then
    log "installing torch-scatter for torch $TORCH_VERSION+$TC_ACCEL"
    nice -n 10 "$PY" -m pip install -q torch-scatter \
      -f "https://data.pyg.org/whl/torch-$TORCH_VERSION+$TC_ACCEL.html"
  fi
fi

log "pip install -e $TC_TREE (no deps)"
nice -n 10 "$PY" -m pip install -q --no-deps -e "$TC_TREE"

log "verify"
"$PY" - <<'EOF'
import importlib, sys
import torch, torch_geometric, lightning, wandb, torchcell
print("python", sys.version.split()[0])
print("torch", torch.__version__, "cuda", torch.version.cuda, "xpu", hasattr(torch, "xpu"))
print("torch_geometric", torch_geometric.__version__)
print("lightning", lightning.__version__, "wandb", wandb.__version__)
print("torchcell", torchcell.__file__)
spec = importlib.util.find_spec("torch_scatter")
print("torch_scatter", "present" if spec else "absent")
EOF
log "done: $PY"
