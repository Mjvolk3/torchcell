#!/bin/bash
# experiments/027-betaxanthin-metabolic/scripts/gh_wave11_ess_size.slurm
# [[experiments.027-betaxanthin-metabolic.wave11-model-size]]
#
# WAVE 11: the MODEL-SIZE LADDER on gene essentiality, pooled arm, no flux layer.
#
# WHY. The wave-9 essentiality curves are violently unstable: on the three pooled cells the
# validation AUROC moves by a median of 0.034, 0.057 and 0.072 BETWEEN ADJACENT EPOCHS,
# against a full range of about 0.3. Seed 2702 peaks at 0.845 at epoch 29 and is selected at
# 0.578. A curve that moves that far in one epoch is not choosing between models, it is
# resampling noise, and the selected epoch is then an order statistic of that noise rather
# than a model property.
#
# THE SIZE ARITHMETIC. The backbone carries 958,501 trainable parameters (622,439 of them in
# the embedding preprocessor) and this target supervises about 926 genotypes, of which 75 are
# validation, so the model has roughly a thousand parameters per training example. The 90-wide
# backbone was inherited from 019's expression lineage, where the supervised set is four
# orders of magnitude larger. Nothing about that transfer was ever checked on this target.
#
# THE LADDER, three widths against the wave-9 baseline at 90, all pooled, all 80 epochs,
# seeds 2700 to 2702:
#
#   hidden 90, 9 heads, 2 layers, preproc 2   the wave-9 baseline (jobs 2009, 2021, 2022)
#   hidden 54, 9 heads, 2 layers, preproc 2   0.6x the width (54, not 45: the perturbation
#                                             head has 6 cross-attention heads and 45 is
#                                             not a multiple of 6; jobs 2039-2041 died on it)
#   hidden 24, 8 heads, 1 layer,  preproc 1   a quarter of the width and one block
#   hidden 12, 4 heads, 1 layer,  preproc 1   the floor
#
# THE PREPROCESSOR DEPTH IS IN THE LADDER BECAUSE THE WIDTH ALONE BARELY MOVES THE SIZE.
# Measured on the 2-epoch smoke (job 2037): dropping the width from 90 to 24 cuts the
# transformer blocks 27-fold, from 196,740 parameters to 7,224, but the total only from
# 958,501 to 568,246, because the two-layer embedding preprocessor's first Linear is
# input_dim by (input_dim + hidden)/2 and does not shrink with the width. It is 64.9 percent
# of the 90-wide model and 96.9 percent of the 24-wide one. At depth 1 it becomes a single
# input_dim by hidden projection, so the two lower rungs are genuinely small models rather
# than the same model with a smaller transformer.
#
# WHAT IT DECIDES. The endpoint is NOT the test AUROC, which at 31 essential genes among 195
# has a bootstrap interval about 0.17 wide and cannot separate these cells. It is the
# STABILITY of the validation curve: the median absolute epoch-to-epoch step, and the spread
# of the selected epoch across seeds. If the small cells hold a flat curve and the large one
# does not, the instability is capacity against a 900-example target and the ladder says so.
# If every width is equally spiky, the cause is elsewhere (the binary target's 16 percent
# positive rate, the batch size of 128 against 851 training rows) and the ladder rules
# capacity out. Both outcomes are worth the GPU hours; neither is assumed here.
#
# GRADIENT TRACKING IS ON for these cells (added 2026-09-15): the pre-clip global gradient
# norm per epoch as a mean and a max, the fraction of batches that hit the clip at 10.0, and
# the norm per top-level module. A spiky validation curve with a flat gradient norm and an
# unstable one are different diagnoses, and until now nothing logged said which this was.
#
# PARAMETERIZED BY ENVIRONMENT, because nine near-identical files would drift apart:
#
#   ESS_HIDDEN=54 ESS_HEADS=9 ESS_LAYERS=2 ESS_SEED=2700 \
#       bash experiments/027-betaxanthin-metabolic/scripts/freeze_and_submit.sh \
#       experiments/027-betaxanthin-metabolic/scripts/gh_wave11_ess_size.slurm
#
# MEMORY. 12g, against a measured peak resident set of 5.4 GB on the wave-10 cells (cgroup
# memory.peak on jobs 2026, 2027 and 2030, 2026-09-15). The node holds 513 GB and the
# database build reserves 256 GB, so four of these plus two CPU audits fit inside the
# remaining 257 GB with room to spare. Requesting 60g, as the frozen snapshots of the
# wave-9 launchers still do, reserves eleven times what the job touches and blocks the node.
#
#SBATCH -p main
#SBATCH --mem=12g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --job-name=027-w11-ess-size
#SBATCH --time=06:00:00
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

HIDDEN="${ESS_HIDDEN:?set ESS_HIDDEN, one of 90 54 24 12; must be a multiple of BOTH head counts}"
HEADS="${ESS_HEADS:?set ESS_HEADS; the width must divide by it}"
LAYERS="${ESS_LAYERS:-2}"
PREPROC="${ESS_PREPROC:-2}"
SEED="${ESS_SEED:-2700}"
EPOCHS="${ESS_EPOCHS:-80}"
echo "host $(hostname) gpu $CUDA_VISIBLE_DEVICES; hidden $HIDDEN heads $HEADS layers $LAYERS preproc $PREPROC seed $SEED"

python experiments/027-betaxanthin-metabolic/scripts/train_bx.py \
    --arm pooled \
    --seed "$SEED" --epochs "$EPOCHS" --graph-reg \
    --head essentiality \
    --hidden "$HIDDEN" --attention-heads "$HEADS" --layers "$LAYERS" \
    --preprocessor-layers "$PREPROC" \
    --wandb-group gh_wave11_esssize --tag "_w11esssize_h${HIDDEN}p${PREPROC}" \
    --out "wave11_ess_pool_h${HIDDEN}l${LAYERS}p${PREPROC}_e${EPOCHS}_s${SEED}.json"
