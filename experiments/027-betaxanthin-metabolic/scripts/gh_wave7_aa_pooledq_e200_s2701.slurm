#!/bin/bash
# experiments/027-betaxanthin-metabolic/scripts/gh_wave7_aa_pooledq_e200_s2701.slurm
# [[experiments.027-betaxanthin-metabolic.wave7-amino-acid-concentration]]
#
# WAVE 7 amino-acid cell: POOLED control, seed 2701.
#
# WHAT WAVE 7 ASKS. Wave 6 measured that the mechanistic amino-acid readout, which reads the
# TURNOVER of each amino acid's GEM metabolite off the predicted flux vector, costs 0.039
# mean per-feature Spearman against the pooled control at three seeds. The anchored
# thermodynamic layer already learns a per-strain log concentration for every metabolite,
# and Mulleder 2016 reports ABSOLUTE intracellular concentrations, so the concentration is
# the quantity the data actually measures. Wave 7 supervises it directly.
#
# THIS CELL IS THE CONTROL, SCORED WITH THE PINBALL LOSS. No metabolic module: the
# amino-acid head is the pooled vector head over the transformer representation, but the
# distributional loss is the 019 quantile grid (K = 19, tau 0.05 to 0.95) instead of the
# Gaussian CRPS, and the point estimate is then the predictive MEDIAN. 019 measured the
# pinball loss as its best distributional loss on expression; whether that carries to the
# amino acids is what this cell measures, paired by seed against the CRPS pooled cell.
# The flux readouts stay on CRPS: their prediction IS the layer's own quantity, so a
# quantile fan would have to come from a separate parameterized head and would break the
# tie the mechanistic arm exists to make.
#
# NOT ON THE WAVE-6 FROZEN TREE. Wave 6 pinned $DATA_ROOT/frozen-027-wave6 because
# flux_layer.py was being edited under the queue. Wave 7 instead runs the code snapshot
# freeze_and_submit.sh takes per job, so every wave-7 cell carries the SHIPPED verified kcat
# table (97 measured / 782 homolog / 2,831 predicted / 18 default) rather than wave 6's
# interim tables. Submit with:
#
#   bash experiments/027-betaxanthin-metabolic/scripts/freeze_and_submit.sh \
#       experiments/027-betaxanthin-metabolic/scripts/gh_wave7_aa_pooledq_e200_s2701.slurm
#
#SBATCH -p main
#SBATCH --mem=12g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --job-name=027-w7e2-aa-pooledq-s2701
#SBATCH --time=10:00:00
#SBATCH --output=/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/kinetics-equilibrator-datasets/experiments/027-betaxanthin-metabolic/slurm/output/%x_%j.out
#SBATCH --error=/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/kinetics-equilibrator-datasets/experiments/027-betaxanthin-metabolic/slurm/output/%x_%j.out

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/kinetics-equilibrator-datasets}"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate torchcell
set -a; . "$PROJECT_ROOT/.env"; set +a
export EXPERIMENT_ROOT="$PROJECT_ROOT/experiments"
cd "$PROJECT_ROOT" || exit 1
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS=ignore
export WANDB_DIR=/scratch/projects/torchcell-scratch/experiments/027-betaxanthin-metabolic/wandb
export WANDB_PROJECT=torchcell_027_bxfx
export NUM_WORKERS=2
export COLLAPSE_PATIENCE_EPOCHS=200
export TC_TEST_PRED_DIR=$DATA_ROOT/test-predictions-027-wave7
mkdir -p "$WANDB_DIR"
echo "host $(hostname) gpu $CUDA_VISIBLE_DEVICES; DATA_ROOT=$DATA_ROOT; EXPERIMENT_ROOT=$EXPERIMENT_ROOT"
EPOCHS="${WAVE7_EPOCHS:-200}"

python experiments/027-betaxanthin-metabolic/scripts/train_bx.py \
    --arm pooled --seed 2701 --epochs "$EPOCHS" --crps --graph-reg \
    --head mulleder19 --dist quantile \
    --wandb-group gh_wave7_aa --tag _w7aa \
    --out wave7_aa_pooledq_e200_s2701.json
