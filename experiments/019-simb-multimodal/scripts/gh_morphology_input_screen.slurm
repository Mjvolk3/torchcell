#!/bin/bash
# experiments/019-simb-multimodal/scripts/gh_morphology_input_screen.slurm
# [[experiments.019-simb-multimodal.scripts.gh_morphology_input_screen]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/gh_morphology_input_screen.slurm
#
# The ridge screen of every served per-deletion measurement against the moving CalMorph
# features (morphology_input_screen.py), under slurm on GilaHyper: it walks the Hillenmeyer
# and Hoepfner LMDBs (about three million records each) read-only.
#
#   sbatch experiments/019-simb-multimodal/scripts/gh_morphology_input_screen.slurm
#SBATCH -p main
#SBATCH --mem=64g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=019-morph1w4-input-screen
#SBATCH --time=08:00:00
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/scratch/projects/torchcell-scratch/experiments/019-simb-multimodal/slurm/output/%x_%j.out
#SBATCH --error=/scratch/projects/torchcell-scratch/experiments/019-simb-multimodal/slurm/output/%x_%j.out
set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate torchcell
cd "${SLURM_SUBMIT_DIR:?submit from the repository root}"
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
python experiments/019-simb-multimodal/scripts/morphology_input_screen.py
