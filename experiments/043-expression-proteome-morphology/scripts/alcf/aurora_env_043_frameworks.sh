#!/bin/bash
# experiments/043-expression-proteome-morphology/scripts/alcf/aurora_env_043_frameworks.sh
# [[experiments.043-expression-proteome-morphology.scripts.alcf.aurora_env_043_frameworks]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/043-expression-proteome-morphology/scripts/alcf/aurora_env_043_frameworks.sh
#
# The matched-stack variant of aurora_env_043.sh (round aurora_2026_10_25 stack test): the
# ALCF frameworks/2026.1.0 module (torch 2.13.0a0+gitcf30153 built against oneAPI 2026.1,
# GPU user-mode driver module intel_gpu_umd_aicoe/2026.06.19 with IGC 2.34.4) plus a venv
# layered on it with --system-site-packages that adds only what the 043 trainer imports and
# the module lacks (torch_geometric, hydra-core, wandb, lmdb, ...; no torch). NO
# LD_LIBRARY_PATH override: the module's own SYCL / Unified Runtime / UMF libraries are the
# point, against the pip torch 2.11.0+xpu env, whose libsycl.so.8 (oneAPI 2025.3) resolves
# libur_loader.so.0 from the default oneapi/2026.1 module through LD_LIBRARY_PATH.
# torchcell is read from the synced tree through PYTHONPATH (its pyproject requires Python
# >= 3.13; the module ships 3.12.12, so it is not pip-installed into the venv).
#
# Rebuild of the venv (login node, once):
#   module load frameworks/2026.1.0
#   python -m venv --system-site-packages /flare/GenesisHackathonOct26/mjvolk3/venvs/tc-frameworks-2026.1
#   then install each module a dry import of train_043.py reports missing, at the version the
#   pip env holds, with --no-deps (pip list of the venv: venv_pip_list_043_frameworks.txt).
#
# Sets STACK=frameworks (aurora_run_cell_043.sh writes it to wandb.stack) and defines
# PY, TRAIN, TC_TREE and the git shim exactly as aurora_env_043.sh does.
TC_PREFIX=/flare/GenesisHackathonOct26/mjvolk3
TC_TREE=$TC_PREFIX/torchcell
VENV=$TC_PREFIX/venvs/tc-frameworks-2026.1
TRAIN=$TC_TREE/experiments/043-expression-proteome-morphology/scripts/train_043.py

# lmod's init and the conda activation the frameworks module runs read unset variables
case $- in *u*) _tc_nounset=1 ;; *) _tc_nounset= ;; esac
set +u
module load frameworks/2026.1.0
# shellcheck disable=SC1091
source "$VENV/bin/activate"
[ -n "$_tc_nounset" ] && set -u
PY=$VENV/bin/python
export STACK=frameworks

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
