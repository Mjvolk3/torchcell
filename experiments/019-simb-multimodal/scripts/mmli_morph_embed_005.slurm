#!/bin/bash
# experiments/019-simb-multimodal/scripts/mmli_morph_embed_005.slurm
# IGB `mmli` MORPHOLOGY arm of the _005 NODE-EMBEDDING (cold-start) study -- the PRIMARY arm.
#
# _003 measured TRAIN per-feature Pearson 0.66 (max 0.78) vs VAL 0.023 (max 0.075): the model
# fits training strains BETTER than the assay's own 0.61 noise ceiling and transfers 3.5%.
# Regularization was inert (dropout, weight decay) while capacity HELPED (hidden 64->128
# doubled val) -- the signature of COLD START, not overfitting. Every prior 019 config ran
# `node_embeddings: []`, making each gene a free learnable vector with no content features,
# so a val gene (never seen perturbed) had nothing linking it to a phenotype.
#
# This sweep gives genes real content features (codon / CaLM / FUDT / ProtT5 / ESM2 / NT,
# plus a random_100 identity control) x learnable_embedding on/off.
#
# ONE mmli node (4 GPUs) -> 4 Optuna workers, each PINNED to one GPU, all pulling from ONE
# shared SQLite study. IGB compute nodes are OFFLINE -> W&B offline, `wandb sync` from a
# LOGIN node afterwards. Self-resubmits until the deadline.
#
#   ssh mjvolk3@biologin.igb.illinois.edu \
#   'sbatch /home/a-m/mjvolk3/projects/torchcell/experiments/019-simb-multimodal/scripts/mmli_morph_embed_005.slurm'
#
#SBATCH -p mmli
# 120g, NOT 488g. compute-5-7 has 500 GB total; requesting ~488 GB demands essentially the
# WHOLE node, so the job cannot schedule whenever another user holds any memory (observed:
# 256 GB allocated -> our job sat PENDING/Resources with an estimated start a YEAR out).
# The 488g came from the old empty-node sniff sweep. 4 single-GPU workers on the 1,439-strain
# fig3_core build need far less; this matches the gh/cabbi launchers.
#SBATCH --mem=120g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --job-name=019-morph-embed005
#SBATCH --time=5-00:00:00
#SBATCH --signal=B:USR1@180
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/home/a-m/mjvolk3/scratch/torchcell/experiments/019-simb-multimodal/slurm/output/%x_%j.out
#SBATCH --error=/home/a-m/mjvolk3/scratch/torchcell/experiments/019-simb-multimodal/slurm/output/%x_%j.out

module purge
source ~/.bashrc
module load singularity

PROJECT_ROOT=/home/a-m/mjvolk3/projects/torchcell
SCRATCH_ROOT=/home/a-m/mjvolk3/scratch/torchcell
OPTUNA_DIR=$SCRATCH_ROOT/experiments/019-simb-multimodal/optuna
mkdir -p "$OPTUNA_DIR"
mkdir -p "$SCRATCH_ROOT/experiments/019-simb-multimodal/slurm/output"
mkdir -p "$SCRATCH_ROOT/experiments/019-simb-multimodal/wandb"

# Run until this wall-clock deadline, resubmitting across walltime boundaries.
export DEADLINE="2026-07-29T10:00:00"
source "$PROJECT_ROOT/experiments/019-simb-multimodal/scripts/requeue_until.sh"
requeue_arm_trap "$PROJECT_ROOT/experiments/019-simb-multimodal/scripts/mmli_morph_embed_005.slurm"

cd "$PROJECT_ROOT/experiments" || exit 1
export NUM_WORKERS=$((SLURM_CPUS_PER_TASK / 4))

singularity exec --nv "$PROJECT_ROOT/rockylinux_9.sif" bash -c "
source \$HOME/miniconda3/bin/activate
conda activate torchcell

export PYTHONPATH=\"$PROJECT_ROOT:\$PYTHONPATH\"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS=ignore

# IGB compute is OFFLINE -> W&B offline, synced from a login node later.
export WANDB_MODE=offline
export WANDB_DIR=$SCRATCH_ROOT/experiments/019-simb-multimodal
export WANDB__SERVICE_WAIT=600

export CONDITION=morph
export JOINT_BASE_CONFIG=cgt_embed_005
export OPTUNA_STORAGE='sqlite:///$OPTUNA_DIR/optuna_019_morph_005.db'
export OPTUNA_STUDY_NAME=morph_005
export OPTUNA_N_TRIALS=\${OPTUNA_N_TRIALS:-30}
export WANDB_PROJECT_SUFFIX=v5
export NUM_WORKERS=$NUM_WORKERS

SWEEP=$PROJECT_ROOT/experiments/019-simb-multimodal/scripts/optuna_joint_sweep.py

# Pre-create the study in ONE process so the 4 workers only load_if_exists — each worker
# spends >15s importing torch before create_study, so all four would otherwise race the
# fresh-DB DDL ('table studies already exists'). Idempotent on resubmit.
python \$SWEEP --create-only

CUDA_VISIBLE_DEVICES=0 OPTUNA_WORKER_ID=0 python \$SWEEP &
CUDA_VISIBLE_DEVICES=1 OPTUNA_WORKER_ID=1 python \$SWEEP &
CUDA_VISIBLE_DEVICES=2 OPTUNA_WORKER_ID=2 python \$SWEEP &
CUDA_VISIBLE_DEVICES=3 OPTUNA_WORKER_ID=3 python \$SWEEP &
wait
" &
# `wait` on the singularity job in the BACKGROUND so the batch shell stays responsive to
# SIGUSR1 (a foreground child would otherwise swallow the signal until it exits).
wait

# Clean exit before the deadline (e.g. all workers finished their trial budget): keep going.
requeue_if_before_deadline "$PROJECT_ROOT/experiments/019-simb-multimodal/scripts/mmli_morph_embed_005.slurm"
