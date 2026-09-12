#!/bin/bash
# experiments/027-betaxanthin-metabolic/scripts/gh_wave7_ff_boxforce_k100_s2702.slurm
# [[experiments.027-betaxanthin-metabolic.wave7-flux-force-head]]
#
# WAVE 7 (2026-09-11): the flux-force channel, everything on, constraint scale 0.0, seed
# 2702, union_external deltaG, the shipped verified kcat table (97/782/2831/18).
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
# THE DIRECTION IN THE BOX FROM THE START, WHICH IS THE FIRST SETTING THAT PASSES BOTH
# HALVES OF THE GENESIS CRITERION AT INITIALIZATION.
#
# Job 1755 and job 1761 (the `force` end, 200 epochs, seeds 2700 and 2701) hold the second
# law on 100 of 100 top strains and sustain their escape, and both still fail Genesis on the
# balance half: medians 1.05 and 0.48 against a tolerance of 1e-2. The probe that says why,
# untrained on the real GEM: after the alternation the flux is in ker S to a median ratio of
# 0.0000, but 0.550 of reactions are outside the box and 0.513 run uphill, so the alternation
# has not reached the intersection and the direction clip that follows buys exactness by
# throwing the balance away.
#
# `boxforce` tightens the box to the licensed direction BEFORE the alternation, so every clip
# in the loop is already the directional one and the balance projection converges against the
# set the flux ends in. Measured balance ratio median after the projection: 0.193 at K = 20,
# 0.0218 at K = 50, 0.0000 at K = 100, against the clip-last end's 0.403, 0.181, 0.057; and on
# the real GEM at K = 100 with the trained-cell recipe, initialization reads balance median
# 0.0017, second-law violation 0.000, box violation 0.000, sign flips 0.000, and
# d(flux sum)/d(log_c_base) = 0.679. The sign clamp `tdg` has the same geometry and NO
# gradient at all, which is what this arm separates.
#
# The cost is density: 70.7 percent of reactions carry no flux at initialization, against the
# force end's 64.5 and the sign clamp's 68.0, and the clamp is the arm that never sustained an
# escape. Whether this one does is the question.
#
# EIGHTY EPOCHS, matching the wave-6 arms rather than job 1755's 200.
#
#   bash experiments/027-betaxanthin-metabolic/scripts/freeze_and_submit.sh \
#       experiments/027-betaxanthin-metabolic/scripts/gh_wave7_ff_boxforce_k100_s2702.slurm
#
#
# 200 EPOCHS, not 80. Job 1745 ran this arm at seed 2700 for 80 epochs and its validation
# curve was still rising at the end (val_r +0.28 at epoch 79 against the 5-epoch moving
# average maximum of 0.235 at the selected epoch 75), so 80 under-trains it. The scoring
# rule is unchanged: the pinned test Spearman at the epoch the centered 5-epoch moving
# average of the validation Pearson selects, never a maximum over epochs.
#
# MEASURED AT SEED 2700 (job 1767, 80 epochs): the Genesis key reads 1.00, the first arm in
# either wave to meet both halves on all 100 top strains (balance ratio median 0.0, second-law
# violation 0.0, sign flips 0.000), and the cell never left the mean plateau: no escape at all
# (val_pred_sd_ratio peaked at 0.0679), pinned test Spearman -0.0312. That is the sign clamp's
# failure mode on an arm that HAS the concentration gradient, so the gradient is not what the
# clamp cost. These seeds are the replication of both halves of that.
#SBATCH -p main
#SBATCH --mem=60g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --job-name=027-w7-ff-boxforce-k100-s2702
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
# The channel. 026's ARMS registry stays the source of truth for the physics switches; the
# parameterization string is passed through by train_bx.py and recorded by the layer in
# flux_meta.coverage.parameterization_raw.
export TC_FLUX_PARAMETERIZATION=fluxforce_k100_boxforce
mkdir -p "$WANDB_DIR"
echo "host $(hostname) gpu $CUDA_VISIBLE_DEVICES; DATA_ROOT=$DATA_ROOT; channel=$TC_FLUX_PARAMETERIZATION"
EPOCHS="${WAVE7_EPOCHS:-80}"

python experiments/027-betaxanthin-metabolic/scripts/train_bx.py \
    --arm flux_projection_anchored_fluxforce --seed 2702 --epochs "$EPOCHS" --crps --graph-reg \
    --kcat-tier predicted --thermo-source union_external \
    --constraint-warmup-epochs 20 --constraint-scale 0.0 --head-normalize \
    --wandb-group gh_wave7_ff --tag _w7ff \
    --out wave7_ff_fluxforce_k100_boxforce_e80_s2702.json
