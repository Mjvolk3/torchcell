#!/bin/bash
# experiments/019-simb-multimodal/scripts/gh_eval_ckpt_predictions.slurm
# [[experiments.019-simb-multimodal.scripts.gh_eval_ckpt_predictions]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/gh_eval_ckpt_predictions
#
# Score saved best-validation checkpoints of the v13 (expression) and v14 (proteome) rounds
# on GilaHyper and dump per-gene validation and test predictions
# (`trainer.eval_ckpt_path`, train_cgt_multitask.py). The checkpoints were copied from IGB
# ($DATA_ROOT/models/checkpoints/<group>/<job>-best-metric-epoch=<e>.ckpt) into the same
# path on GilaHyper; the dataset, split cache and embeddings are the GilaHyper builds the
# rounds were mirrored from. Each task rebuilds the run's arm through gh_expr_008_arm.sh so
# the config and partition are the run's own; the eval prints the val metrics, which must
# match the W&B run at the checkpoint's epoch.
#
# Submit from the worktree root:
#   sbatch --array=0-5 experiments/019-simb-multimodal/scripts/gh_eval_ckpt_predictions.slurm
#
#SBATCH -p main
#SBATCH --gres=gpu:1
#SBATCH --mem=64g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=019-eval-ckpt
#SBATCH --time=06:00:00
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/scratch/projects/torchcell-scratch/experiments/019-simb-multimodal/slurm/output/%x_%A_%a.out
#SBATCH --error=/scratch/projects/torchcell-scratch/experiments/019-simb-multimodal/slurm/output/%x_%A_%a.out

set -euo pipefail
mkdir -p /scratch/projects/torchcell-scratch/experiments/019-simb-multimodal/slurm/output
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -d "$SLURM_SUBMIT_DIR/torchcell" ]; then
  PROJECT_ROOT="$SLURM_SUBMIT_DIR"
else
  echo "FATAL: submit from a torchcell checkout" >&2
  exit 1
fi
source ~/miniconda3/etc/profile.d/conda.sh
conda activate torchcell
export PYTHONUNBUFFERED=1
export WANDB_MODE=disabled
CK=/scratch/projects/torchcell-scratch/models/checkpoints
TASK="${SLURM_ARRAY_TASK_ID:?submit with --array}"

# task: config, arm, seed, checkpoint (W&B run in the comment)
case "$TASK" in
  0) CFG=cgt_expr_v13_split;    ARM=V_ref_s0;    SEED=0
     CKPT="$CK/compute-0-2-2397312_3fcfb279cc58c21f926585737901556cf40cded9e6c3fc039c1b34abbc9b7248/best_metric.ckpt" ;;  # wq8y8nd5
  1) CFG=cgt_expr_v13_split;    ARM=V_concat_s0; SEED=0
     CKPT="$CK/compute-0-2-2397312_ecd690dcabb6286e9a2f646c3232e14167f234c7778433803e0420024f402c61/best_metric.ckpt" ;;  # bn37i9vs
  2) CFG=cgt_expr_v13_split;    ARM=V_ref_s2;    SEED=0
     CKPT="$CK/compute-0-1-2397316_f4f3eb97310e9f0ded0ac8f43fac8569a5fa44a5d023576aeac414c734472433/best_metric.ckpt" ;;  # lp6guytz
  3) CFG=cgt_expr_v14_proteome; ARM=P_ref_s0;    SEED=0
     CKPT="$CK/compute-3-3-2400351_45b093533290bd592f8cfc6d59a1f07556b3c4aa16eb56882545a02c020b8845/best_metric.ckpt" ;;   # uc0pm2pv
  4) CFG=cgt_expr_v14_proteome; ARM=P_concat_s0; SEED=0
     CKPT="$CK/compute-3-3-2400351_190566f3daabdfe4dc73d7309269139c074951adb6e07764d3234aa0b20a9247/best_metric.ckpt" ;;   # bc8bngdr
  5) CFG=cgt_expr_v14_proteome; ARM=P_ref_s2;    SEED=1
     CKPT="$CK/compute-3-3-2401108_e4609a5c2e63232a6035fab136b7875f37697ad44dfd33241d871e14adddd99b/best_metric.ckpt" ;;    # 25ok3tce
  *) echo "unknown task $TASK" >&2; exit 1 ;;
esac
[ -f "$CKPT" ] || { echo "FATAL: missing checkpoint $CKPT" >&2; exit 1; }
echo "task $TASK: $CFG $ARM seed $SEED"
echo "checkpoint: $CKPT"

export TORCHCELL_CONFIG="$CFG"
export TORCHCELL_EXTRA_OVERRIDES="trainer.eval_ckpt_path=$CKPT trainer.run_test=false data_module.num_workers=4"
bash "$PROJECT_ROOT/experiments/019-simb-multimodal/scripts/gh_expr_008_arm.sh" "$ARM" "$SEED"
