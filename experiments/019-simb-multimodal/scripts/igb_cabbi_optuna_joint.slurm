#!/bin/bash
# experiments/019-simb-multimodal/scripts/igb_cabbi_optuna_joint.slurm
# IGB `cabbi` CONTROLLED expr<->morph joint sweep — runs on IGB (proven torch2.8/PyG2.7 env)
# to sidestep the Delta env rebuild. cabbi = compute-3-3 (8x RTX6000 24GB); this grabs 4.
#
# Runs the THREE conditions SEQUENTIALLY in one job (only 4 GPUs free — parallel jobs would
# just queue). Each: pre-create the Optuna study, then 4 GPU-pinned workers, then wait.
#   morph / expr = single-objective; joint = MULTI-objective (Pareto of expr x morph).
# All on the SAME 1,440 both-modality instances (require_modalities), so joint-vs-baseline
# isolates the auxiliary-task effect. IGB compute is OFFLINE -> W&B offline + `wandb sync` later.
#
#   sbatch experiments/019-simb-multimodal/scripts/igb_cabbi_optuna_joint.slurm
#   CONDITIONS="joint" sbatch --export=ALL,CONDITIONS=joint .../igb_cabbi_optuna_joint.slurm   # one only
#
#SBATCH -p cabbi
#SBATCH --mem=120g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --job-name=019-joint-cabbi
#SBATCH --time=20:00:00
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

cd "$PROJECT_ROOT/experiments" || exit 1
export NUM_WORKERS=$((SLURM_CPUS_PER_TASK / 4))

# Conditions to run, in order (override: --export=ALL,CONDITIONS="joint").
CONDITIONS="${CONDITIONS:-morph expr joint}"

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

export JOINT_BASE_CONFIG='delta_joint_expr_morph_000'
export OPTUNA_N_TRIALS=20
export NUM_WORKERS=$NUM_WORKERS

SWEEP=$PROJECT_ROOT/experiments/019-simb-multimodal/scripts/optuna_joint_sweep.py

for COND in $CONDITIONS; do
  echo \"===== CONDITION=\$COND =====\"
  export CONDITION=\$COND
  export OPTUNA_STORAGE=\"sqlite:///$OPTUNA_DIR/optuna_019_joint_\${COND}_peak.db\"
  export OPTUNA_STUDY_NAME=\"joint_\${COND}_peak_000\"

  # Pre-create the study (one process) so the 4 workers only load_if_exists — kills the
  # fresh-DB DDL race; --create-only sets single- vs multi-objective directions correctly.
  python \$SWEEP --create-only

  CUDA_VISIBLE_DEVICES=0 OPTUNA_WORKER_ID=0 python \$SWEEP &
  CUDA_VISIBLE_DEVICES=1 OPTUNA_WORKER_ID=1 python \$SWEEP &
  CUDA_VISIBLE_DEVICES=2 OPTUNA_WORKER_ID=2 python \$SWEEP &
  CUDA_VISIBLE_DEVICES=3 OPTUNA_WORKER_ID=3 python \$SWEEP &
  wait
  echo \"===== DONE CONDITION=\$COND =====\"
done
"
