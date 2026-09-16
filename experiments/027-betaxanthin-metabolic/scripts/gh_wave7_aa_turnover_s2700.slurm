#!/bin/bash
# experiments/027-betaxanthin-metabolic/scripts/gh_wave7_aa_turnover_s2700.slurm
# [[experiments.027-betaxanthin-metabolic.wave7-amino-acid-concentration]]
#
# WAVE 7 amino-acid cell: TURNOVER readout, everything on, seed 2700.
#
# THE ARM UNDER TEST. `flux_projection_anchored` (mass balance exact by construction,
# anchored thermodynamics on the union_external table, the predicted-kcat capacity bound
# through the abundance head, the protein-pool budget and the dissipation limit) at
# constraint scale 0.1 over a 20-epoch warmup, with `--aa-readout flux_concentration`.
#
# WHAT THE READOUT IS. Wave 6's `flux_metabolite` head: a_k log(1 + omega_k) + b_k on the
# TURNOVER of each amino acid's GEM metabolite, two free parameters per column. It is rerun
# here rather than read off wave 6 because wave 6's cells ran a different kcat table, a
# different arm (thermodynamics off) and the shipped concentration window, so they are not a
# controlled comparator for the concentration cells. `--head-normalize` divides the initial
# slope by log1p(flux_scale), which is wave 6's measured fix for this head's epoch-0 offset.
#
# WHY THE WINDOW IS WIDENED. The anchored concentration is squashed into
# [conc_lower_m, conc_upper_m], shipped as 1e-7 to 1e-2 M. Mulleder's measured intracellular
# amino acids run to 9.5e-2 M and 10.6 percent of all 88,882 measurements are above 1e-2 M,
# with glutamate and glutamine above it in 99.8 percent of strains, so under the shipped
# ceiling the supervised value would be unreachable and the sigmoid would saturate silently
# (mulleder_concentration_window.py, results/mulleder_concentration_window.json). The
# ceiling is raised to 1 M, a decade above the largest measurement, for EVERY wave-7 flux
# cell including the turnover arm, so the turnover-versus-concentration contrast is on the
# readout alone and not on the window.
#
# NOT ON THE WAVE-6 FROZEN TREE: freeze_and_submit.sh snapshots the code per job, so this
# cell carries the shipped verified kcat table (97 measured / 782 homolog / 2,831 predicted
# / 18 default), which every wave-7 result file records under flux_meta.kcat_tier_counts.
#
#   bash experiments/027-betaxanthin-metabolic/scripts/freeze_and_submit.sh \
#       experiments/027-betaxanthin-metabolic/scripts/gh_wave7_aa_turnover_s2700.slurm
#
#SBATCH -p main
#SBATCH --mem=12g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --job-name=027-w7-aa-turn-s2700
#SBATCH --time=05:00:00
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
EPOCHS="${WAVE7_EPOCHS:-80}"

python experiments/027-betaxanthin-metabolic/scripts/train_bx.py \
    --arm flux_projection_anchored --kcat-tier predicted --thermo-source union_external \
    --constraint-warmup-epochs 20 --constraint-scale 0.1 \
    --seed 2700 --epochs "$EPOCHS" --crps --graph-reg \
    --head mulleder19 --head-normalize --aa-readout flux_metabolite --conc-upper-m 1.0 \
    --wandb-group gh_wave7_aa --tag _w7aa \
    --out wave7_aa_turnover_s2700.json
