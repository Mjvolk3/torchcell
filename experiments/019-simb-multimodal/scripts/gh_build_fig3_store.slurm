#!/bin/bash
# experiments/019-simb-multimodal/scripts/gh_build_fig3_store.slurm
# [[experiments.019-simb-multimodal.scripts.gh_build_fig3_store]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/gh_build_fig3_store.slurm
#
# Build a Figure 3 store from the served graph on GilaHyper, under slurm (DB work never
# runs as a login-shell process). The tag selects the query and the dataset root:
#   fig3_morph_proteome  the three expression panels + Messner 2023 + Ohya 2005, the store
#                        the proteome-revealed morphology arms of v24 need (2026-10-10)
#   fig3_proteome        the 2026-09-15 build, for a rebuild
# Then mirror to IGB: bash experiments/019-simb-multimodal/scripts/sync_igb_fig3_core.sh <tag>
#
#   sbatch --export=ALL,FIG3_TAG=fig3_morph_proteome \
#       experiments/019-simb-multimodal/scripts/gh_build_fig3_store.slurm
#SBATCH -p main
#SBATCH --mem=96g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=019-build-fig3-store
#SBATCH --time=12:00:00
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/scratch/projects/torchcell-scratch/experiments/019-simb-multimodal/slurm/output/%x_%j.out
#SBATCH --error=/scratch/projects/torchcell-scratch/experiments/019-simb-multimodal/slurm/output/%x_%j.out
set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate torchcell
cd "${SLURM_SUBMIT_DIR:?submit from the repository root}"
export PYTHONPATH="$PWD"
TAG="${FIG3_TAG:?set FIG3_TAG=fig3_morph_proteome|fig3_proteome}"
echo "host $(hostname)  tag $TAG  commit $(git rev-parse --short HEAD)  $(date)"
python experiments/019-simb-multimodal/scripts/build_fig3_proteome.py --tag "$TAG"
echo "done $(date)"
