#!/bin/bash
# experiments/027-betaxanthin-metabolic/scripts/gh_wave7_ff_force_e200_s2701.slurm
# [[experiments.027-betaxanthin-metabolic.wave7-flux-force-head]]
#
# WAVE 7 (2026-09-11): the flux-force channel, everything on, constraint scale 0.0, seed
# 2701, union_external deltaG, the shipped verified kcat table (97/782/2831/18).
#
# THE SECOND LAW AS A SMOOTH COUPLING RATHER THAN A SIGN CLAMP. The directional box
# (`_tdg`) decides each reaction's allowed sign with `delta_g > eps` inside a torch.where,
# which is a step function: no gradient reaches the learned log concentrations through the
# box, so the model cannot move a metabolite pool to license a direction along that path.
# Measured in wave 6: every projection cell WITHOUT the sign tightening sustained its escape
# from the betaxanthin mean plateau (6 of 6, jobs 1689/1690/1738/1734/1735/1739) and none
# with it did (0 of 6, jobs 1723/1724/1725/1730/1731/1726).
#
# Here the direction and a magnitude factor come from the reaction energy through the
# flux-force relation (Beard and Qian; the flux-force efficacy of Noor et al. 2014),
#
#     v_j = m_j f(dG_j),  f(x) = (1 - e^{x/RT}) / (1 + e^{x/RT}) = -tanh(x / 2RT),
#     m_j = max(|lb_j|, |ub_j|) sigmoid(z_j) >= 0,
#
# so the sign is structural, |v_j| <= m_j, a near-equilibrium reaction goes quiet instead of
# being pinned, and dv/dln c is nonzero everywhere. The flux is then clipped to the genotype
# box, which is what zeroes an irreversible reaction whose energy forbids it, and run
# through the SAME twenty alternations ending on the clip that the tdg and boxend cells run.
#
# THE PAIRED CONTROLS ARE ALREADY IN HAND at these seeds and on this kcat table:
# flux_projection_anchored_tdg nocstr (jobs 1725/1730/1731) is the sign clamp, and
# flux_projection_anchored_k20_boxend nocstr (jobs 1734/1735/1739) is the same projection
# with no thermodynamics in the box. This arm's switches are the boxend arm's, so the three
# differ in the parameterization string and nothing else.
#
# MEASURED AT INITIALIZATION, untrained, on the real yeast-GEM (batch 4, seed 0): the
# pre-projection flux violates the second law on 0.000 of the priced reactions, and the
# projection then flips the sign of 0.273 of the reactions carrying flux, leaving a
# violation fraction of 0.318 against the plain projection's 0.308. The flip fraction is
# NOT an artifact of the starting magnitude: biasing the magnitude logit by -6, which takes
# the pre-projection mean |v| from 15.8 to 0.7, leaves it at 0.289. `feas_sign_flip_frac` is
# logged every epoch so the trained number is read rather than assumed.
#
# THE DIRECTION RESTORED LAST. Job 1745 measured what the box end costs: the
# pre-projection flux violates the second law on 0.000 of the priced reactions, the balance
# projection then flips 0.2846 of the flux-carrying ones, and the top-100 model violation
# median lands at 0.2851, so the sign-by-construction property did not survive the
# projection and the Genesis key read 0.00. `fluxforce_k20_force` ends the alternation on
# the BALANCE projection and then clips into the box tightened to the direction each
# reaction's energy licenses, so bounds and the second law are both exact and the balance
# residual is the reported leftover.
#
# It is not the `tdg` head. The magnitude and the direction that enter that clip are already
# smooth functions of delta_r_G, so the learned concentrations keep a gradient path; the
# clamp head had none. Measured at initialization on the real yeast-GEM: second-law
# violation 0.000 and box violation 0.000 after the clip, balance ratio median 0.392, 63.8
# percent of reactions at zero flux against the sign clamp's 68.0 percent, and
# d(flux sum)/d(log_c_base) nonzero.
#
# THE DISCRIMINATOR. If the escape survives here, the gradient is what the sign clamp cost.
# If it dies as the clamp's did, the cause is the zeroing, not the gradient.
#
#   bash experiments/027-betaxanthin-metabolic/scripts/freeze_and_submit.sh \
#       experiments/027-betaxanthin-metabolic/scripts/gh_wave7_ff_force_e200_s2701.slurm
#
#
# 200 EPOCHS, not 80. Job 1745 ran this arm at seed 2700 for 80 epochs and its validation
# curve was still rising at the end (val_r +0.28 at epoch 79 against the 5-epoch moving
# average maximum of 0.235 at the selected epoch 75), so 80 under-trains it. The scoring
# rule is unchanged: the pinned test Spearman at the epoch the centered 5-epoch moving
# average of the validation Pearson selects, never a maximum over epochs.
#
# MEASURED AT SEED 2700 (job 1755, 200 epochs): the force end holds the second law on
# 100 of the 100 top predicted strains, exactly as the sign clamp does, AND sustains its
# escape from epoch 17 with a pinned test Spearman of +0.0499 at the selected epoch 185,
# where the clamp sustained 0 of 3 escapes and scored -0.0202 on average. What is still
# open is the balance residual: median 1.05 on the top 100, so the Genesis key reads 0.00.
# These two seeds are the replication of that single cell.
#SBATCH -p main
#SBATCH --mem=60g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --job-name=027-w7-ff-force-e200-s2701
#SBATCH --time=08:00:00
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
# The channel. 026's ARMS registry stays the source of truth for the physics switches; the
# parameterization string is passed through by train_bx.py and recorded by the layer in
# flux_meta.coverage.parameterization_raw.
export TC_FLUX_PARAMETERIZATION=fluxforce_k20_force
mkdir -p "$WANDB_DIR"
echo "host $(hostname) gpu $CUDA_VISIBLE_DEVICES; DATA_ROOT=$DATA_ROOT; channel=$TC_FLUX_PARAMETERIZATION"
EPOCHS="${WAVE7_EPOCHS:-200}"

python experiments/027-betaxanthin-metabolic/scripts/train_bx.py \
    --arm flux_projection_anchored_fluxforce --seed 2701 --epochs "$EPOCHS" --crps --graph-reg \
    --kcat-tier predicted --thermo-source union_external \
    --constraint-warmup-epochs 20 --constraint-scale 0.0 --head-normalize \
    --wandb-group gh_wave7_ff --tag _w7ff \
    --out wave7_ff_fluxforce_k20_force_e200_s2701.json
