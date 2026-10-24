#!/bin/bash
# experiments/043-expression-proteome-morphology/scripts/gh_train_043_ontology_smoke_cpu.slurm
# [[experiments.043-expression-proteome-morphology.scripts.gh_train_043_ontology_smoke_cpu]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/043-expression-proteome-morphology/scripts/gh_train_043_ontology_smoke_cpu.slurm
#
# CPU smoke of the ontology-conditioning arm on prep_v4 (no GPU: the GilaHyper cards are the
# overnight round's). Arm all, HL-Gauss, equal weighting, batch 8, every mark, a few steps per
# source: model.conditioning.source=ontology, then =name on the same code path, then the
# leave-Hu-out arm. Proves shapes, the loader and the evaluation keys before any GPU time.
# W&B offline (nothing uploaded).
#
#   sbatch experiments/043-expression-proteome-morphology/scripts/gh_train_043_ontology_smoke_cpu.slurm
#SBATCH -p main
#SBATCH --mem=96g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=043-onto-smoke-cpu
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/projects/torchcell-scratch/experiments/043-expression-proteome-morphology/slurm/output/%x_%j.out
#SBATCH --error=/scratch/projects/torchcell-scratch/experiments/043-expression-proteome-morphology/slurm/output/%x_%j.out
set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate torchcell
cd "${SLURM_SUBMIT_DIR:?submit from the repository root}"
export PYTHONPATH="$PWD"
export DATA_ROOT=/scratch/projects/torchcell-scratch
export EXPERIMENT_ROOT="$PWD/experiments"
export WANDB_MODE=offline
echo "host $(hostname)  commit $(git rev-parse --short HEAD)  $(date)"
for case in "ontology all []" "name all []" "ontology lodo [hu]"; do
  set -- $case
  echo "===== source $1  arm $2  exclude $3  $(date)"
  python experiments/043-expression-proteome-morphology/scripts/train_043.py \
    data.store_version=v2 data.prep_version=prep_v4 \
    arm=all "arm.exclude_from_training=$3" model.conditioning.source="$1" \
    loss=hlgauss weighting=equal seed=0 device=cpu \
    data.batch_size=8 data.num_workers=4 data.sampler=records data.mark_cap=null \
    data.val_max_records_per_dataset=16 data.traineval_max_records_per_dataset=8 \
    budget.max_epochs=1 +budget.max_steps_per_epoch=6 budget.probe_steps=2 \
    wandb.mode=offline "wandb.tags=[smoke,ontology_2026_10_23]" wandb.round=ontology_smoke_cpu
done
echo "done $(date)"
