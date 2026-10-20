#!/bin/bash
# experiments/040-inhibitor-synergy-wetlab/scripts/gh_build_040_store.slurm
# [[experiments.040-inhibitor-synergy-wetlab.scripts.gh_build_040_store]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/040-inhibitor-synergy-wetlab/scripts/gh_build_040_store
#
# Build the experiment 040 store (the four pooled chemogenomic screens and the private
# Bioscreen wells of KG 4.0, one entry per served record, with a manifest) from the served
# graph on GilaHyper, under slurm: DB work never runs as a login-shell process. Submit
# from the worktree root that holds the experiment:
#
#   sbatch experiments/040-inhibitor-synergy-wetlab/scripts/gh_build_040_store.slurm
#
#   TC040_STORE_TAG   store directory under $DATA_ROOT/data/torchcell/experiments/040-.../
#                     (default mixture_040)
#   TC040_GENES       smoke: gene set separated by spaces or semicolons (sbatch splits
#                     --export on commas); set it in the shell and pass --export=ALL
#   TORCHCELL_SRC     the torchcell checkout to import, default the release's own commit
#                     (build/kg-4-0-v1-8-0, 1.8.3, the package KG 4.0 is paired with)
#
#   smoke:  TC040_GENES="YGL013C YBL005W YDR011W YAL002W YBR020W" TC040_STORE_TAG=mixture_040_smoke \
#             sbatch --export=ALL --time=01:00:00 experiments/040-inhibitor-synergy-wetlab/scripts/gh_build_040_store.slurm
#
# Sizing: the 033 build of the same four screens (6.39M records) ran at 16 CPUs / 64 GB
# (slurm 2929, single session, 7 h 48 min; the partitioned raw stage measured 0.307 ms
# per record on the Hoepfner block, tcdb-002 slurm 3138). Three stage copies of about
# 110 GB each land under the store root; raw/ and aggregation/ are regenerable.
#SBATCH -p main
#SBATCH --mem=64g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=040-build-store
#SBATCH --time=12:00:00
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/scratch/projects/torchcell-scratch/experiments/040-inhibitor-synergy-wetlab/slurm/output/%x_%j.out
#SBATCH --error=/scratch/projects/torchcell-scratch/experiments/040-inhibitor-synergy-wetlab/slurm/output/%x_%j.out
set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate torchcell
cd "${SLURM_SUBMIT_DIR:?submit from the worktree root}"
TORCHCELL_SRC="${TORCHCELL_SRC:-/home/michaelvolk/Documents/projects/torchcell.worktrees/build/kg-4-0-v1-8-0}"
[[ -f "$TORCHCELL_SRC/torchcell/__init__.py" ]] || { echo "ERROR: no torchcell at $TORCHCELL_SRC" >&2; exit 1; }
export PYTHONPATH="$TORCHCELL_SRC"
export TORCHCELL_SRC
export PYTHONUNBUFFERED=1
echo "host $(hostname)  experiment tree $(git rev-parse --short HEAD)  torchcell $(git -C "$TORCHCELL_SRC" rev-parse --short HEAD)  $(date)"
echo "store tag ${TC040_STORE_TAG:-mixture_040}  smoke genes ${TC040_GENES:-none}"
# The build script asserts that torchcell resolved from TORCHCELL_SRC (a `python -c`
# check here would resolve the submit directory's own torchcell first, from cwd).
python experiments/040-inhibitor-synergy-wetlab/scripts/build_040_store.py
echo "done $(date)"
