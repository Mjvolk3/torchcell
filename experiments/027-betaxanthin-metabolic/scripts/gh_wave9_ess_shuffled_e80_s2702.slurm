#!/bin/bash
# experiments/027-betaxanthin-metabolic/scripts/gh_wave9_ess_shuffled_e80_s2702.slurm
# [[experiments.027-betaxanthin-metabolic.wave9-essentiality]]
#
# WAVE 9: gene essentiality, `shuffled`, seed 2702.
#
# THIS CELL IS the SHUFFLED-PARAMETER control from the review: the same architecture and the same parameter count with the kcat table permuted across catalytic units. If the flux arm beats pooled with a permuted kcat table too, what it gained was the architecture and not the measured kinetics.
#
# THE TARGET. SGD gene essentiality, a BINARY label, scored against Merzbacher 2025's own
# released yeast split (223 test genes: 192 nonessential, 31 essential). This is the endpoint
# where the published methods are far enough apart to decide something; the amino-acid target
# has a ceiling near 0.17 Pearson and every arm sits at 0.05 to 0.07.
#
# THE COMPARATORS, and they are the YEAST numbers. From the mirrored si/si1.md Supplementary
# Table S5 (table at line 33, caption line 35): FBA accuracy 0.81, precision 0.91, recall
# 0.97, ROC AUC 0.69; FCL test accuracy 0.89, precision 0.89, recall 0.99, ROC AUC 0.73; FCL
# 5-fold CV ROC AUC 0.78 +/- 0.065. The 93.5 and 95 percent figures quoted elsewhere are the
# E. COLI accuracies (paper.md line 54) and do not apply here.
#
# ROC AUC IS THE HEADLINE, NOT ACCURACY. The majority-class rate on those 223 genes is 0.861
# and FBA's published accuracy is 0.81, i.e. BELOW it. Accuracy cannot separate these methods
# and AUROC can, so the epoch is selected on validation AUROC under the same centered 5-epoch
# moving average the continuous heads use, and the test block reports AUROC, accuracy, and
# precision and recall PER CLASS beside the majority rate.
#
# THE LOSS IS A BINARY CROSS ENTROPY on one logit, and --crps is refused on this head: CRPS
# is a proper scoring rule for a continuous target and a Gaussian over a 0/1 label is not the
# right object. The target is NOT standardized; a class label handed a train-split mean and
# sd would leave the log-loss scoring a rescaled variable that is not a probability's target.
#
# 80 EPOCHS, not 200. Every arm on this recipe that escaped did so inside the first 40 epochs
# (ibpool at 10 to 19, bxpool at 20, bxprec at 40), and the 200-epoch isobutanol cells spent
# 175 epochs overfitting after their selected epoch. The review budgeted 40; 80 is twice that.
#
# THE LABEL IN THE RELEASED SPLIT IS INVERTED and the trainer inverts it once: the column
# named `essential` carries 1.0 for a NONESSENTIAL gene. Counted on the CSV, test=0 holds 772
# at 1.0 and 126 at 0.0, matching the paper's "772 nonessential; 126 essential" with the
# labels reversed, and Supplementary Table S4 agrees at the total (964 nonessential, 157
# essential). Reading it at face value inverts every label, which a classifier fits happily.
#
# BLOCKED AS OF 2026-09-14 AND NOT SUBMITTED. The dataset this cell trains on does not exist
# yet: the served Neo4j store is reachable and EMPTY (0 nodes), so
# gh_cpu_build_essentiality.slurm cannot run. Submit these only after that build lands.
#
# THE MEDIUM IS THE DATASET'S, AND THE MISMATCH IS DELIBERATE AND LABELED. `--apply-media
# dataset` writes rich medium (YPD-approx) onto the GEM's exchange bounds before the layer is
# built, because that is what the SGD essentiality phenotype was scored on. The PUBLISHED FBA
# comparator is a different medium: paper.md line 122 says its baseline is cobrapy
# `single_gene_deletion` "for all genes with glucose as the carbon source and the standard
# biomass reaction", i.e. glucose minimal. So the FBA row and this arm do not share a medium,
# and the result file records both facts rather than presenting the comparison as matched.
# Measured on the GEM: YPD-approx opens 45 exchanges and takes wild-type growth from 0.0858
# to 0.5352 1/h.
#
#   bash experiments/027-betaxanthin-metabolic/scripts/freeze_and_submit.sh \
#       experiments/027-betaxanthin-metabolic/scripts/gh_wave9_ess_shuffled_e80_s2702.slurm
#
#SBATCH -p main
#SBATCH --mem=12g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --job-name=027-w9-ess-shuffled-s2702
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
EPOCHS="${WAVE9_EPOCHS:-80}"

python experiments/027-betaxanthin-metabolic/scripts/train_bx.py \
    --arm flux_projection_anchored --kcat-tier predicted --thermo-source union_external \
    --constraint-warmup-epochs 20 --constraint-scale 0.1 \
    --seed 2702 --epochs "$EPOCHS" --graph-reg --apply-media dataset \
    --head essentiality --conc-space mixture --conc-upper-m 1.0 \
    --flux-shuffle-kcat --wandb-group gh_wave9_ess --tag _w9essshuffled \
    --out wave9_ess_shuffled_e80_s2702.json
