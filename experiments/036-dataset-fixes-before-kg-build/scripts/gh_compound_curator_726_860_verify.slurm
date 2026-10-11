#!/bin/bash
#SBATCH --job-name=036-726w1-verify
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --output=/scratch/projects/torchcell/database/slurm/output/%A_%a_%x.out
#SBATCH --error=/scratch/projects/torchcell/database/slurm/output/%A_%a_%x.out
#SBATCH --no-requeue
# experiments/036-dataset-fixes-before-kg-build/scripts/gh_compound_curator_726_860_verify.slurm
# [[experiments.036-dataset-fixes-before-kg-build.scripts.gh_compound_curator_726_860_verify]]
#
# L0-L4 verification of the dev stores the #726/#860 curator pass touched whose build did
# not verify them itself (the build CLI's `--verify` found no per-dataset entry point, so
# the family runner or a module route the sweep knows covers them), plus the touched
# loaders whose stores were not rebuilt. One array task per line of CLASSES_FILE runs
# kg4_pre_build_verification_sweep.py for that ONE dataset into its own results file, so
# the 7-hour Hoepfner atlas does not serialize the rest; the files are merged afterwards
# with `--merge`. Read-only on processed/; the routes write preprocess reports only.
#
#   sbatch --array=1-18%4 --export=ALL,CLASSES_FILE=<one class per line>,TORCHCELL_SRC=<worktree> \
#       experiments/036-dataset-fixes-before-kg-build/scripts/gh_compound_curator_726_860_verify.slurm
#
# Sizing: the 2026.10.09 sweep took 25,989 s for Hoepfner 2014 and 4,341 s for Price
# 2018 RB-TnSeq, every other store here under 30 s.
set -euo pipefail
echo "Running on $(hostname) at $(date); array task ${SLURM_ARRAY_TASK_ID:?not an array job}"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate torchcell
: "${CLASSES_FILE:?set CLASSES_FILE=<one dataset class name per line>}"
: "${TORCHCELL_SRC:?set TORCHCELL_SRC=<the checkout whose code verifies>}"
cd "$TORCHCELL_SRC"
DATASET_CLASS=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$CLASSES_FILE")
[ -n "$DATASET_CLASS" ] || { echo "no class on line $SLURM_ARRAY_TASK_ID of $CLASSES_FILE"; exit 1; }
OUT_DIR="${OUT_DIR:-/scratch/projects/torchcell/database/slurm/verify_726_860}"
mkdir -p "$OUT_DIR"
echo "source: $TORCHCELL_SRC ($(git rev-parse --short HEAD)); dataset: $DATASET_CLASS"
PYTHONPATH="$TORCHCELL_SRC" python \
    experiments/036-dataset-fixes-before-kg-build/scripts/kg4_pre_build_verification_sweep.py \
    --dataset "$DATASET_CLASS" --out "$OUT_DIR/$DATASET_CLASS.json"
echo "DONE at $(date)"
