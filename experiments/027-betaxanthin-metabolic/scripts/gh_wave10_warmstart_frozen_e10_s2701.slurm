#!/bin/bash
# experiments/027-betaxanthin-metabolic/scripts/gh_wave9_warmstart_genoref_frozen_e200_s2700.slurm
#
# DRIVER 2026-09-15 00:45: job 1946 (alpha learned, single wild-type reference, old gate) met the
# turnover-thresholded rule on 46 of the top 100 at its selected epoch 3 (audit job 1985), then
# grew alpha 0.010 -> 0.087 and lost it (worst-strain max 2.0 by epoch 45). This cell freezes
# alpha at 0.01, keeps the single wild-type reference (the per-genotype pFBA table moves 2 of
# 32 references and MOMA equals the wild type, jobs 1971/1982), and runs under the FIXED gate
# (deletion_indices; ARO4/ARO7 open). Shipped bounds, as 1946. Read: genesis_turnover_rule_1e-03
# on the top 100 at the selected epoch AND the trajectory of the training-batch active max.
# [[experiments.026-metabolism-flux.physics-gates-2026-09-14]]
#
# WAVE 9, E1: THE WARM-STARTED RESIDUAL HEAD.
#
# THE DIAGNOSIS THIS ARM ACTS ON, all of it measured on 2026-09-14 and all of it from
# committed scripts under experiments/026-metabolism-flux/scripts/.
#
#   * The alternating projection cannot repair an O(1)-wrong proposal at ANY K. From a
#     uniform random point of the box, K = 20 to 5,000, von Neumann and Dykstra alike, the
#     turnover-thresholded balance ratio stays at 2.0, the worst value the statistic takes.
#     What K buys is emptiness: 1,808 active rows at K = 20 down to 557 at K = 5,000
#     (results/probe_projrate.json, job 1900).
#   * From a proposal 1 percent from feasible, the EXISTING K = 100 alternation reaches
#     balance_active_max 0.00118 and p95 0.00038 with 82 active rows and a flux-carrying
#     fraction of 0.21 (results/probe_warmstart.json, job 1900). The reference itself reads
#     83 rows and 0.107.
#   * The trained fluxforce_k100_boxforce head of job 1784 proposes at a relative distance
#     of 0.990, i.e. the projection discards 99 percent of what the head emits, over 32
#     genotypes (results/probe_trained_proposal_distance_wave8_dens_d2p0_boxforce_k100_e80_s2700.json,
#     job 1904). The proposal, not the projection, is what fails.
#
# THE CHANNEL. v_pre = clip(v_ref, box) + alpha * ||clip(v_ref)|| * N z / ||N z||, with
# v_ref a pFBA solution of the unperturbed GEM solved inside the layer's build step from S,
# the bounds and the objective column, and alpha one learned scalar initialized at 0.01, so
# the proposal's relative distance from the reference IS alpha. Then the same directional
# tightening and the same K = 100 alternation the boxforce cells run.
#
# DENSITY AND ACTIVE-ROW BALANCE AT 0.0, unlike the wave-8 cells. The control is
# jobs 1767/1770/1771, which ran this recipe at both weights zero, so the contrast is the
# channel and nothing else. The density term was in any case pushing the wrong way: it was
# built to raise the active-row count from 422 and the pFBA reference is 83.
#
# ----------------------------------------------------------------------------------------
# DO NOT SUBMIT WITHOUT READING results/probe_warmstart_init.json (job 1922).
#
# The initialization check on the REAL GEM does not reach the regime the synthetic probe
# predicts. Untrained, batch 4, built through 026's own build_flux_layer on this arm:
# balance_active_max 1.955, p95 1.824, 20 active rows, flux_active_frac 0.208, and the
# projection still discarding 0.929 of the proposal, against the target 0.00118 / 0.00038 /
# 82 / 0.21.
#
# The same file names the cause, and it is not the warm start: the genotype box does not
# contain the reference. The enzyme-capacity ceiling has a median of 0.0485 mmol/gDW/h over
# GPR reactions against a reference whose largest flux is 9.12, so 82 of the reference's 495
# flux-carrying reactions sit above their own ceiling by a median factor of 4.31, and
# clipping costs 0.761 of the reference's norm with NO directional tightening at all. The
# directional tightening then forbids the direction of a further 145 of those 495 reactions
# outright, which is the box-level form of the standard-state result in
# results/probe_dirbox.json (maximum downhill throughput exactly 0.0).
#
#
# THE 2x2 FOLLOW-UP (job 1925) DID NOT RESCUE IT, AND IT NAMED THE REAL BLOCKER. Exempting
# the 47 reactions no concentration can rescue and switching to the nominal E_bar abundance
# path move every statistic the right way and monotonically (best cell: max 2.000, p95 1.616,
# 59 active rows, proposal distance 0.849) and stay three orders of magnitude from the
# balance target. The LP in the same file says why: inside this directionally tightened box
# the maximum turnover any balanced point can carry over the 4,045 one-sided reactions is
# exactly 0.00 without the exemption and 76.76 with it, and the witness has 0 and 14 active
# rows against the reference's 82.
#
# THAT READING WAS ONE STEP SHORT, AND JOB 1933 CORRECTS IT. The tightening is evaluated at
# the layer's own concentrations, which at initialization license nothing in particular.
# Starting them at the LP witness that DOES license the reference (log_c_init_path, written
# by thermo_exempt_gate.py) takes sign-forbidden reference reactions from 124 to 14, makes
# the tightened box feasible as built for the first time, and reads max 0.657, p95 0.276 and
# 120.5 active rows. The tightened box then reaches 98 of the 105 turning-over rows the
# UNtightened box reaches, so the second law in the box costs almost nothing once the
# concentrations are consistent.
#
# WHAT STILL BLOCKS IT IS CAPACITY, in three named reactions carrying 14.8 percent of the
# reference's squared norm: r_0226 (ATP synthase) at 29.8x its enzyme-capacity ceiling,
# r_0439 at 2.61x and r_0961 at 3.06x. That is a kcat or an abundance question, not a head
# question. Run this cell with --arm naming the nominal abundance path, the exemption list
# and the concentration init only after those three are resolved.
# So this cell is READY and is NOT the next thing to run. Two things would have to move
# first, and they have different owners: the abundance prior, which sets the capacity
# ceiling, and the directional tightening, which the E2 exemption list (47 reactions,
# results/thermo_exempt_candidates.json) is the first candidate fix for.
# ----------------------------------------------------------------------------------------
#
#
# ----------------------------------------------------------------------------------------
# THE FULL FLAG SET, PREPARED 2026-09-14 AND NOT SUBMITTED.
#
# Five things had to move before this cell had a premise, and each was measured before it
# was changed. Initialization values are job 1937 unless stated.
#
#   1. NOMINAL ABUNDANCE, via `base_arm: flux_nullspace` in the arm file. At the prior
#      abundance the box admits NO balanced point at all: the reachability LP is infeasible
#      with or without the directional tightening. Nominal takes the capacity ceiling median
#      from 0.0485 to 8.22 mmol/gDW/h and the over-ceiling count from 82 to 3.
#   2. THE EXEMPTION LIST, 47 reactions no concentration in the physiological window can
#      make downhill. Removing them takes the thermodynamic-feasibility LP total L1 slack
#      from 2,641.5 kJ to 0.000 over the remaining 378 constraints (job 1906).
#   3. THE CONSISTENT CONCENTRATIONS, the same LP witness. Starting the learned log c there
#      takes the reference reactions the tightening forbids from 124 to 14, makes the
#      tightened box feasible as built for the first time, and takes bal_act_max from 2.000
#      to 0.657 (job 1929).
#   4. THE NORMALIZED AVAILABILITY FOLD. The softmin returns 1 - log(n_u)/beta on an INTACT
#      complex, so a 17-subunit complex reads 0.646 on a wild type. Dividing by that value
#      raises ATP synthase box ceiling from 4.4707 to 6.9222, exactly the 1.548 the bias
#      predicts, and holds the reaction (job 1937).
#   5. THE KCAT WIDENING at 2 sigma from the committed tier ladder, which is the only rule
#      that covers all three over-ceiling reactions: no subunit-aggregation rule reaches
#      r_0961, a single-gene unit. It is the sourced rule rather than a convention swap.
#
# WHERE IT LANDS AT INITIALIZATION, and this cell is an experiment rather than a pass:
#
#   bal_act_max 0.0306, p95 0.0019, 81.8 active rows, density 0.1774, proposal distance
#   0.018, against the target 0.00118 / 0.00038 / 82 / 0.21 / 0.008.
#
# THE p95 HALF OF THE RULE IS ALREADY MET AT INIT AND THE MAXIMUM IS 3.1x SHORT. The row
# count is at target, every watched reaction is held, and the clip costs 0.0073 of the
# reference norm, so the residual is not the reference failing to fit the box. What this
# cell tests is whether the PRICED balance term closes the remaining factor now that the
# geometry admits a balanced flux at all. That is a hypothesis, not a pre-approved pass.
#
# PRE-REGISTERED READ, from the review rank 1:
#   (a) `genesis_turnover_rule_1e-03` on the top 100 at the selected epoch, success >= 0.95
#       on >= 2 of 3 seeds. The selected epoch is the centered 5-epoch moving average of the
#       validation Pearson, never a maximum over epochs.
#   (b) `flux_active_frac` and `balance_active_rows_1e-03` must move TOWARD 0.107 and 83,
#       not away. Init reads 0.1774 and 81.8, so the density starts BELOW target here and
#       moving it up is as much a failure as moving it down was in wave 8.
#   (c) pinned test Spearman must not fall more than the control seed sd (0.076) below the
#       control mean (+0.056).
#
# KILL, verbatim from the review: "rule not met on any seed -> the warm start is not the
# lever; go to E2/E3."
# CONTINGENCY, verbatim: "if it passes but the score collapses: the head is pinned to
# v_ref. Diagnose with across-genotype flux sd; if near zero, ramp alpha up over training
# and rerun one seed."
#
# TWO FLAGS BELOW DO NOT EXIST YET, and this cell is not submittable until they do:
#   --flux-reference-table      -> FluxLayerConfig.warm_start_reference_table_path
#   --flux-alpha-frozen         -> FluxLayerConfig.warm_start_alpha_learnable = False
#   --flux-warm-start-init-rel  -> FluxLayerConfig.warm_start_init_rel
# The patches follow the pattern --flux-parsimony-weight established and are written out in
# [[experiments.026-metabolism-flux.physics-gates-2026-09-14]]. The rest exist: the
# parsimony flag landed at 03:50, the exempt, log-c and availability flags at 05:47.
# The k_cat rule is NOT a trainer flag: it rides in `base_arm: flux_nullspace_kcat_sigma2`,
# which 026's build_flux_layer forwards to parameters.resolve_kcat_table as
# kcat_ceiling_sigma_k=2.0. ArmConfig forbids extra keys, so the registry entry is the only
# channel an 027 arm file has for it, and `base_arm` is serialized with the cell.
# ----------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------
# WAVE 9b: THE REFERENCE CARRIES THE GENOTYPE.
#
# WHAT JOB 1946 MEASURED, read at epoch 143 of 200. The single-reference warm start never
# left the betaxanthin mean plateau: validation prediction spread 0.0004 to 0.012,
# validation Pearson flat at -0.052 after epoch 5, training data loss flat at 0.527 from
# epoch 0. Meanwhile the learned alpha grew from 0.0099 to 0.087, the training-batch
# active-row balance max went from about 1.0 to 2.0 by epoch 45, c_balance_active from
# 0.001 to 0.17, p95 swinging 0.005 to 0.9 after epoch 110, and density 0.175 to 0.25. The
# box held at 0 violation and g_diss stayed at 1,314 to 1,568 J/gDW/h, under the 3,700
# limit it was never penalized against.
#
# THE DIAGNOSIS THIS CELL TESTS, and it is a hypothesis rather than a finding: with one
# wild-type reference the proposal is v_ref + alpha N z, so the flux is very nearly the same
# vector for every genotype, the head reading it is close to a constant predictor, and the
# only channel the data term has for genotype information is alpha. Growing alpha is exactly
# what it did, and feasibility was the price.
#
# ONE MORE NUMBER THAT POINTS THE SAME WAY. The training-batch active-row max was already
# about 1.0 at epoch 0 while the init probe read 0.03, and the init probe forwarded four
# WILD TYPES. Deletion genotypes fail the rule from the start, which no wild-type-only
# check could see.
#
# WHAT CHANGES. `--flux-reference-table` supplies one pFBA solution PER GENOTYPE, keyed by
# the sorted deleted-gene set, solved with the genes knocked out through the GPR by
# cobra.manipulation.knock_out_model_genes and cached under $DATA_ROOT with its provenance.
# A genotype the table does not carry falls back to the wild-type reference clipped into its
# own box, and the fallback rate is reported by coverage_report rather than assumed.
#
# THE READ IS THE ACROSS-GENOTYPE FLUX SD as much as the balance rule. A head that is
# feasible and still emits one vector for every genotype has not fixed anything, and that
# statistic is what separates the two.
# ----------------------------------------------------------------------------------------
#
# THIS CELL: ALPHA FROZEN AT 0.02, no gradient. If the reference now carries the genotype,
# the residual does not have to, and a fixed small residual keeps the proposal inside the
# regime where the K = 100 alternation reaches the balance rule. If this arm trains and the
# learned-alpha sibling does not, the residual was the problem; if neither does, the
# reference is still not carrying enough and the diagnosis above is wrong.
#
#   bash experiments/027-betaxanthin-metabolic/scripts/freeze_and_submit.sh \
#       experiments/027-betaxanthin-metabolic/scripts/gh_wave9_warmstart_genoref_frozen_e200_s2700.slurm
#
#SBATCH -p main
#SBATCH --mem=60g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --job-name=027-w10-frz-e10-s2701
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
# THE RESOLVED KCAT TABLE DEPENDS ON THE HASH SEED. Measured by the kinetics owner: 2 of
# 3,728 values move by up to 0.313 decades between interpreter runs, so every kinetics
# script now refuses to run without this and a cell that omitted it would not be
# reproducible from its own record.
export PYTHONHASHSEED=0
export WANDB_DIR=/scratch/projects/torchcell-scratch/experiments/027-betaxanthin-metabolic/wandb
export WANDB_PROJECT=torchcell_027_bxfx
export NUM_WORKERS=2
export COLLAPSE_PATIENCE_EPOCHS=200
export TC_TEST_PRED_DIR=$DATA_ROOT/test-predictions-027-wave7
mkdir -p "$WANDB_DIR"
echo "host $(hostname) gpu $CUDA_VISIBLE_DEVICES; DATA_ROOT=$DATA_ROOT; channel from arm_flux_warmstart_boxforce.yaml"
EPOCHS="${WAVE9_EPOCHS:-10}"
SEED="${WAVE9_SEED:-2701}"
RESULTS_026="$PROJECT_ROOT/experiments/026-metabolism-flux/results"
EXEMPT_PATH="$RESULTS_026/thermo_exempt_candidates.json"
LOG_C_PATH="$RESULTS_026/thermo_consistent_log_c.json"

python experiments/027-betaxanthin-metabolic/scripts/train_bx.py \
    --arm flux_warmstart_boxforce_nominal \
    --flux-density-weight 0.0 --flux-balance-active-weight 0.5 \
    --flux-parsimony-weight 1e-3 \
    --flux-thermo-exempt-path "$EXEMPT_PATH" \
    --flux-log-c-init-path "$LOG_C_PATH" \
    --flux-availability-fold normalized_softmin \
    --flux-alpha-frozen --flux-warm-start-init-rel 0.01 \
    --seed "$SEED" --epochs "$EPOCHS" --crps --graph-reg \
    --kcat-tier predicted --thermo-source union_external \
    --constraint-warmup-epochs 20 --constraint-scale 0.0 --head-normalize \
    --wandb-group gh_wave9_warmstart --tag warmstart_frozen_e10 \
    --out "wave10_warmstart_frozen_e10_k100_boxforce_e${EPOCHS}_s${SEED}.json"
