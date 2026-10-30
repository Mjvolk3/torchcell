#!/bin/bash
# Builds the torchcell043 env on the CPU node's local disk the way
# experiments/042-hpc-smoke-tests/scripts/install_env.sh builds the ALCF envs (python 3.13,
# torch 2.11.0+cu128 from pytorch.org, env/requirements.txt, torch-scatter from the PyG wheel
# index), then packs it into ONE tarball on /scratch (Lustre metadata is ~60 ms per small-file
# create; jobs unpack it to node-local /tmp, the 040 torchcell-r11.tar pattern). The full
# GilaHyper freeze is not used: it is not resolvable as a set (job 23299939, click==8.3.3
# ResolutionImpossible). A constraints file holds torch, PyG and numpy at GilaHyper's versions.
# torchcell itself is NOT installed: jobs import it from the tree through PYTHONPATH.
#SBATCH -A bfjt-delta-cpu
#SBATCH -p cpu-interactive
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32g
#SBATCH --time=01:00:00
#SBATCH --job-name=043-envbuild
#SBATCH --output=/scratch/bbub/mjvolk3/torchcell-043/env_build/%x_%j.out
#SBATCH --error=/scratch/bbub/mjvolk3/torchcell-043/env_build/%x_%j.out
set -euo pipefail
T=/scratch/bbub/mjvolk3/torchcell-043
B=$T/env_build
L=/tmp/043envbuild_${SLURM_JOB_ID}
TARBALL=/scratch/bbub/mjvolk3/envs/torchcell043.tar
mkdir -p "$L"
trap 'rm -rf "$L"' EXIT
export HOME=$L/home PIP_CACHE_DIR=$L/cache/pip TMPDIR=$L/cache/tmp XDG_CACHE_HOME=$L/cache/xdg
export MAMBA_ROOT_PREFIX=$L/cache/mamba
mkdir -p "$HOME" "$PIP_CACHE_DIR" "$TMPDIR" "$XDG_CACHE_HOME"
log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
log "host $(hostname); local disk: $(df -h /tmp | tail -1)"
ENV=$L/torchcell043
PY=$ENV/bin/python
log "micromamba create python 3.13.0 -> $ENV"
/scratch/bbub/mjvolk3/envs/bin/micromamba create -y -q -p "$ENV" -c conda-forge --override-channels "python=3.13.0" pip
log "python: $($PY --version)"
printf 'torch==2.11.0\ntorch_geometric==2.8.0\nnumpy==2.2.6\n' > "$L/constraints.txt"
log "torch 2.11.0 (cu128)"
"$PY" -m pip install -q "torch==2.11.0" --index-url https://download.pytorch.org/whl/cu128
log "torch: $($PY -c 'import torch; print(torch.__version__)')"
log "env/requirements.txt (constraints: torch 2.11.0, torch_geometric 2.8.0, numpy 2.2.6)"
"$PY" -m pip install -q -r "$T/env/requirements.txt" -c "$L/constraints.txt" \
  --extra-index-url https://download.pytorch.org/whl/cu128
log "torch-scatter 2.1.2 for torch 2.11.0+cu128"
"$PY" -m pip install -q "torch-scatter==2.1.2" -c "$L/constraints.txt" \
  -f https://data.pyg.org/whl/torch-2.11.0+cu128.html
log "torch after all installs: $($PY -c 'import torch; print(torch.__version__)')"
"$PY" -m pip check || true
"$PY" -m pip freeze > "$B/delta_freeze.txt"
log "pack -> $TARBALL"
tar cf "$TARBALL.partial" -C "$L" torchcell043
mv "$TARBALL.partial" "$TARBALL"
ls -la "$TARBALL"
log "verify from a relocated unpack, importing from the Delta tree"
mkdir -p "$L/verify" && tar xf "$TARBALL" -C "$L/verify"
cd "$T"
PYTHONPATH="$T" "$L/verify/torchcell043/bin/python" - <<'PYEOF'
import sys, torch, torch_geometric, torch_scatter, h5py, numpy, wandb, hydra, pydantic
print("python", sys.version.split()[0], sys.prefix)
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("torch_geometric", torch_geometric.__version__, "torch_scatter", torch_scatter.__version__)
print("numpy", numpy.__version__, "h5py", h5py.__version__, "wandb", wandb.__version__, "pydantic", pydantic.__version__)
import torchcell, torchcell.models.cell_model_043 as m
print("torchcell", torchcell.__file__)
print("cell_model_043", m.__file__)
PYEOF
sha256sum "$TARBALL" > "$TARBALL.sha256"
cat "$TARBALL.sha256"
log "done"
