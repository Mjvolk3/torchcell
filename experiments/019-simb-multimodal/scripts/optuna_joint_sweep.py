# experiments/019-simb-multimodal/scripts/optuna_joint_sweep.py
# [[experiments.019-simb-multimodal.scripts.optuna_joint_sweep]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/optuna_joint_sweep
"""OPTIMIZER x DISTRIBUTIONAL Optuna sweep driver (_007) for the Fig-3 multitask CGT.

WHAT THE PREVIOUS ROUND SETTLED. _006 took expression from 0.109 to 0.1746 peak per-feature
Pearson by porting 010's NINE-graph configuration (9 graphs, 9 attention heads, one
regularized head per graph), with 0 failed trials. Architecture is therefore FROZEN here:
hidden=90, L=4, 9 heads (see cgt_expr_007.yaml). Against the replicate-based ceiling of
0.775, expression is at ~23% of achievable -- far from saturated, so there is real headroom
for this round to find.

WHAT IT LEFT OPEN, and what this round spends its budget on:

* OPTIMIZER (the big one). _006's largest single measured effect was `hp_profile`, a ~2x
  spread between two BUNDLES of (lr, dropout, weight_decay). A bundle is unattributable, and
  the naive reading is actively suspect: it says weight_decay 1e-4 helps, while 010's
  checkpointed production model used 1e-8. _007 decomposes the bundle into three independent
  axes, two of them CONTINUOUS log-uniform.

* DISTRIBUTIONAL (`multitask.dist`). Five proper scoring rules -- crps, quantile,
  laplace_crps, nll, energy -- replacing _006's {point, crps, quantile}. `point` is dropped
  (its optimum IS the conditional mean, i.e. the mean-collapse the metric punishes).

* JOINT vs MARGINAL (`multitask.energy_rank` in {0, 32}). Every other loss scores each gene
  independently; `energy` scores the whole 6,169-gene vector. Holding the loss at `energy`
  and varying only the rank of the covariance factor isolates the joint question. GATED on
  measurement: residual_covariance_diagnostic.py found the residual gene-gene correlation
  reproducible (split-half r = 0.869 vs a shuffled null of 0.0001) with effective rank 32.8.

* GRAPH ABLATION (`graph_reg_on`, `graph_reg_depth`). _006 ran lambda > 0 in every trial, so
  "graphs help" rests on a cross-round comparison against runs that also differed in head
  count, embeddings, and a since-fixed double-lambda bug. A within-round lambda=0 arm settles
  it; early-vs-middle injection tests an assumption inherited from 010 and never checked.

RANKING is on the PEAK of `val/<phenotype>/pearson_per_feature` -- the honest metric
(per-feature across strains, destroyed by mean-collapse), computed from `DistHead.point()`
so every distributional arm is compared on the same point-estimate correlation. Peak, not
last epoch, because these runs peak early then collapse toward the per-feature mean.

CALIBRATION is recorded alongside but does NOT enter the objective: `calib/coverage_50`,
`calib/coverage_80` (cross-mode comparable) and `calib/pit_ks` (within-mode only), read at
the peak epoch. Ranking on them would be wrong -- a model that predicts the marginal
distribution and ignores the input is perfectly calibrated and useless -- but a
distributional sweep that never looks at them is not measuring what it claims to.

The structural decoder axis (`s1_pool` vs `s3_xattn`) is retained for arms with the
morphology head; expression is per-token by construction (S0) and has no decoder lever.

THE CONTROL: all three conditions run on the SAME instance set (`require_modalities` in the
base config), so expr_morph - morph isolates the auxiliary-task effect from a data-quantity
confound. Same split as _002, so _003 is directly comparable.

    CONDITION=morph      active_heads=[global]           single-obj: morphology
    CONDITION=expr       active_heads=[per_gene]         single-obj: expression (S0 control)
    CONDITION=expr_morph active_heads=[per_gene, global] MULTI-obj: (expr, morph) Pareto

Environment (set by the slurm launchers):
    CONDITION           expr | morph | expr_morph   (default: expr_morph)
    OPTUNA_STORAGE      sqlite:////<scratch>/.../optuna_019_<cond>_007.db   (required)
    OPTUNA_STUDY_NAME   default: <condition>_007
    OPTUNA_N_TRIALS     trials THIS worker runs (default: 20)
    OPTUNA_WORKER_ID    seeds the Halton scramble (default: 0) -- give each worker a
                        DIFFERENT id so the six of them cover complementary sub-cubes
    JOINT_BASE_CONFIG   base Hydra config (default: cgt_expr_007)
    WANDB_PROJECT_SUFFIX  W&B project suffix (default: v7)
    NUM_WORKERS         dataloader workers per trial (default: 4)
"""

import os
import os.path as osp
import sys

import optuna
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from train_cgt_multitask import run_training

CONF_DIR = osp.abspath(osp.join(osp.dirname(__file__), "../conf"))
BASE_CONFIG = os.getenv("JOINT_BASE_CONFIG", "cgt_expr_007")
STORAGE = os.environ["OPTUNA_STORAGE"]
CONDITION = os.getenv("CONDITION", "expr_morph")
STUDY_NAME = os.getenv("OPTUNA_STUDY_NAME", f"{CONDITION}_007")
N_TRIALS = int(os.getenv("OPTUNA_N_TRIALS", "20"))
WORKER_ID = int(os.getenv("OPTUNA_WORKER_ID", "0"))
PROJECT_SUFFIX = os.getenv("WANDB_PROJECT_SUFFIX", "v7")

ACTIVE_HEADS = {
    "expr": ["per_gene"],
    "morph": ["global"],
    "expr_morph": ["per_gene", "global"],
}[CONDITION]

# PEAK (max over training), not the last epoch — runs peak then MSE-collapse to the mean.
# Namespaced by PHENOTYPE (stable across decoder variants), computed from DistHead.point()
# so the ranking is loss-agnostic.
EXPR_METRIC = "val/expression/pearson_per_feature_max"
MORPH_METRIC = "val/morphology/pearson_per_feature_max"

# ---- Distributional axis, round _007 ----
# `point` (MSE) is DROPPED. Its optimum IS the conditional mean, which is precisely the
# mean-collapse the honest per-feature metric punishes; _003 measured that and the sweep does
# not need to re-measure it. The five survivors are all proper scoring rules, so each is
# minimized in expectation by the TRUE predictive distribution -- they differ in the shape
# they assume and in what they penalize:
#   crps         -- closed-form Gaussian CRPS. In y-units, no 1/sigma^2 term, so no collapse.
#   quantile     -- pinball over K=19 evenly spaced tau. Distribution-FREE: no shape assumed.
#   laplace_crps -- closed-form Laplace CRPS. Same machinery as `crps` with heavier tails;
#                   the test of whether the Gaussian tail assumption is costing anything.
#   nll          -- Gaussian NLL. Included as the KNOWN-PATHOLOGICAL control: it can drive
#                   sigma -> 0 on well-fit rows for unbounded reward. The calib/* metrics are
#                   what make that visible (n-shaped PIT, coverage ABOVE nominal) instead of
#                   inferring it from a diverging loss.
#   energy       -- multivariate CRPS. The only JOINT member: it scores the whole 6,169-gene
#                   vector at once and can therefore be paid for representing gene-gene
#                   covariance. Gated on the residual diagnostic (see ENERGY_RANKS).
DIST_CHOICES = ["crps", "quantile", "laplace_crps", "nll", "energy"]

# ---- Energy rank axis (the JOINT ablation) ----
# Sigma = diag(sigma^2) + V V^T with V of rank k. Varying ONLY k while holding the loss at
# `energy` isolates "does modelling the joint pay?" from "is the energy score a better
# objective?" -- comparing energy against crps would confound the two, since they differ in
# both. k=0 makes V None: same loss, same sampler, independent genes.
#
# k=32 is MEASURED, not guessed. residual_covariance_diagnostic.py finds the residual
# gene-gene correlation structure REPRODUCIBLE (split-half r = 0.869 against a
# within-gene-shuffled null of 0.0001, so it is not sampling noise) with a participation-
# ratio effective rank of 32.8. That measurement is the gate: had the split-half r sat at
# the null, there would have been nothing for a joint head to learn and this axis would not
# exist.
ENERGY_RANKS = [0, 32]

# ---- Graph-regularization injection depth ----
# With num_transformer_layers FROZEN at 4 (layers 0..3), [1] is early and [2] is middle.
# _006 used [1] for every trial and never tested the alternative, so "early is right" is
# currently an untested inheritance from 010 rather than a finding.
GRAPH_REG_LAYERS = {"early": "[1]", "middle": "[2]"}

# `raw` is GONE as a lever: every vector head is standardized (see the base config's
# require_standardized_targets). An un-standardized 278-D CalMorph target makes any y-unit
# loss (MSE/CRPS/pinball) chase a few large-magnitude features while the metric weights all
# 278 equally — that would confound the decoder comparison.
TARGET_NORMS = ["zscore", "yeo_johnson"]

# ---- Node-embedding (cold-start) axis, round _005 ----
# Every option is COVERAGE-AUDITED: all 6607 genes, 0 missing, 0 zero-vectors. Variants that
# fail that bar are deliberately absent (one_hot_gene / fudt_downstream miss the 28
# mitochondrial Q0* genes; the esm2/prot_T5 `no_dubious` / `no_uncharacterized` variants emit
# 684/668 ALL-ZERO vectors, i.e. silently information-free for the hardest ORFs).
#   (""            -- DROPPED: learnable-only baseline, superseded by random_100.)
#   codon_frequency (64d)  -- only option smaller than hidden, so no lossy compression.
#   calm (768d), fudt_upstream (768d), prot_T5_all (1024d), esm2_..._all (1280d),
#   nt_window_5979 (2560d) -- real sequence/protein content.
#   random_100 (100d)      -- CONTROL: unique but meaningless. Paired with
#     learnable_embedding=false it isolates CONTENT from IDENTITY; with learnable=true it is
#     redundant with the free table, which is why `learnable` is swept alongside it.
# NOTE: "" (no content features) is DROPPED now that learnable_embedding is pinned False
# -- "" + no free table leaves genes with no node features at all. `random_100` is the
# strictly better version of that baseline: same "unique but meaningless" semantics, but
# explicit, fixed, and usable at cold start.
#
# ROUND _007 CHANGES:
#   * `random_100` -> `random_1000`. The control exists to answer "how much of this
#     embedding's benefit is CONTENT rather than a unique-vector-plus-graph-position
#     fingerprint?", and that only works if the control is comparable in WIDTH. At 100
#     dimensions it was 7-25x narrower than every real embedding it was being compared
#     against (calm/fudt 768, prot_T5 1024, esm2 1280, nt_window 2560), so a win for a real
#     embedding was partly a win for having more dimensions. 1000d sits in the middle of
#     that range and makes the comparison about content.
#   * `fudt_upstream,fudt_downstream` (concatenated, 1536d) added. fudt is the PROMOTER +
#     TERMINATOR axis -- regulatory context, orthogonal to the protein-coding axis that
#     prot_T5/esm2 cover -- and only the upstream half was ever in the sweep. Downstream is
#     usable for the first time this round: its artifact was missing the 28 mitochondrial
#     Q0* genes until backfill_fudt_downstream.py rebuilt it (6,607 ids, 0 missing).
#     A comma-joined entry becomes a MULTI-embedding override, which the model concatenates.
NODE_EMBEDDINGS = [
    "codon_frequency",
    "calm",
    "fudt_upstream",
    "fudt_upstream,fudt_downstream",
    "prot_T5_all",
    "esm2_t33_650M_UR50D_all",
    "nt_window_5979",
    "random_1000",
]


def _norm_choice(trial: optuna.Trial, head: str) -> str:
    """Per-modality target-norm lever, always standardized ({zscore, yeo_johnson})."""
    param = "expr_norm" if head == "per_gene" else "morph_norm"
    return trial.suggest_categorical(param, TARGET_NORMS)


def objective(trial: optuna.Trial) -> float | tuple[float, float]:
    # ---- Decoder study axes ----
    # `decoder` is only meaningful when the morphology (global) head is active; expression
    # is S0 per-token by construction. Suggesting it for the expr arm would add a phantom
    # dimension that splits the TPE search space without changing the model.
    if "global" in ACTIVE_HEADS:
        decoder = trial.suggest_categorical("decoder", ["s1_pool", "s3_xattn"])
    else:
        decoder = "s1_pool"
    # The joint arm drops `quantile` (K=19 params x 6127 genes x 278 features is a large
    # readout on a 4-GPU shared node); morph/expr sweep the full set.
    dist_choices = (
        [d for d in DIST_CHOICES if d != "quantile"]
        if CONDITION == "expr_morph"
        else DIST_CHOICES
    )
    dist = trial.suggest_categorical("dist", dist_choices)
    # Suggested for EVERY trial, not only energy ones. Optuna's search space is defined by
    # which params a trial calls, so suggesting conditionally would make the space ragged --
    # and QMC in particular assumes a fixed-dimension unit cube. The value is inert unless
    # dist == energy, and it is recorded either way so a W&B/Optuna filter never hits a null.
    energy_rank = trial.suggest_categorical("energy_rank", ENERGY_RANKS)

    # ---- Cold-start axes (_005) ----
    node_emb = trial.suggest_categorical("node_embeddings", NODE_EMBEDDINGS)
    # learnable_embedding is NO LONGER SWEPT -- it is pinned False.
    #
    # Measured on the actual split (seed 0, the seed these runs use): only 26 of 539
    # validation genes (4.8%) and 14 of 519 test genes (2.7%) are ever perturbed in
    # training. Because every strain is a SINGLE deletion, splitting by strain IS
    # splitting by gene -- there is no warm-start regime in this dataset. So a free
    # per-gene row is still at initialization for ~95% of evaluation genes and cannot
    # contribute to the ranked (validation) metric; it can only memorize train, which is
    # exactly the train 0.9 / val 0.02 gap observed on morphology.
    #
    # Sweeping it therefore spent trials on a lever that provably cannot move val. The
    # honest version of "unique but meaningless vector" is `random_100`, which stays in
    # NODE_EMBEDDINGS as the explicit control -- and, combined with graph regularization,
    # is also the test of whether GRAPH POSITION alone carries the signal (a fixed random
    # vector makes each gene's neighborhood aggregate a computable fingerprint even for an
    # unseen gene).
    learnable = False

    # hidden_channels and num_transformer_layers are NO LONGER SWEPT -- both are frozen in
    # cgt_expr_007.yaml (90 and 4). _006 swept them and neither moved the result the way
    # `hp_profile` did, so this round spends the freed budget on the optimizer instead.

    # ---- Graph-regularization axes ----
    # THE ABLATION. `graph_reg_on = false` sets lambda to exactly 0, which is the only
    # honest test of whether the graph prior is carrying _006's 0.109 -> 0.1746 gain. Every
    # _006 trial had lambda > 0, so "graphs help" was inferred by comparing ACROSS rounds --
    # against runs that also differed in head count, embedding set and the double-lambda
    # bug. A within-round lambda=0 arm removes all of that.
    graph_reg_on = trial.suggest_categorical("graph_reg_on", [False, True])
    # MEASURED, not guessed. calibrate_graph_reg_lambda.py ran one epoch per lambda on
    # THIS config (9 graphs, single regularized layer, main's per-graph edge-count
    # normalization) and read val/graph_reg/ratio_to_data = graph_term / data_term:
    #
    #     lambda   1e-8      1e-6      1e-4      1e-2
    #     ratio    0.00048   0.048     4.17      406
    #
    # so ratio ~ 4.8e4 * lambda and PARITY (ratio = 1) sits at lambda ~ 2e-5. This grid
    # spans ratio ~0.01 -> ~10, centred on parity. Suggested unconditionally (fixed-
    # dimension space, as with energy_rank) and then zeroed when the ablation is on.
    graph_reg_grid = trial.suggest_categorical("graph_reg_lambda", [2e-7, 2e-6, 2e-5, 2e-4])
    graph_reg = graph_reg_grid if graph_reg_on else 0.0
    reg_depth = trial.suggest_categorical("graph_reg_depth", list(GRAPH_REG_LAYERS))

    # ---- Optimizer axes: the hp_profile bundle, DECOMPOSED ----
    # `hp_profile` was _006's largest single effect (~2x) and its least interpretable: it
    # moved lr, dropout and weight_decay together, so the win could belong to any of the
    # three. The reading is not even obvious -- the naive one says weight_decay 1e-4 helps,
    # but 010's checkpointed production model used 1e-8. Separating them is the point of
    # this round.
    #
    # lr and weight_decay are CONTINUOUS log-uniform, which is what makes the Halton sampler
    # worth using: QMC fills a continuous cube with low discrepancy, but falls back to
    # independent random sampling for categoricals. Their ranges are set to CONTAIN both
    # profiles and 010's setting rather than to bracket a guess:
    #     lr           5e-5 .. 2e-3   (spans baseline 3e-4, aggressive 1e-3, 010's 1e-4)
    #     weight_decay 1e-9 .. 1e-3   (spans aggressive 1e-4 and 010's 1e-8, and goes low
    #                                  enough to be effectively off)
    lr = trial.suggest_float("lr", 5e-5, 2e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-9, 1e-3, log=True)
    # Categorical, not continuous: dropout is a coarse regularization dial and the interesting
    # question is "any vs none", not the third decimal. 0.3 is excluded -- both _006 profiles
    # sat at or below 0.1 and nothing suggested the model is over-fitting hard enough to want it.
    dropout = trial.suggest_categorical("dropout", [0.0, 0.1, 0.2])

    # Build the two per-head normalization lists from each active head's sampled lever.
    normalize_list: list[str] = []   # -> Yeo-Johnson (vector_norm_method)
    standardize_list: list[str] = []  # -> per-feature z-score
    for head in ACTIVE_HEADS:
        choice = _norm_choice(trial, head)
        if choice == "zscore":
            standardize_list.append(head)
        else:
            normalize_list.append(head)

    overrides = [
        f"multitask.active_heads=[{','.join(ACTIVE_HEADS)}]",
        # node_emb may be a COMMA-JOINED multi-embedding (e.g. fudt upstream+downstream);
        # it is interpolated raw so the list override carries both names.
        f"cell_dataset.node_embeddings=[{node_emb}]",
        f"model.learnable_embedding.enabled={learnable}",
        f"multitask.decoder={decoder}",
        f"multitask.dist={dist}",
        f"multitask.energy_rank={energy_rank}",
        f"multitask.normalize_vector_targets=[{','.join(normalize_list)}]",
        f"multitask.standardize_per_feature_target=[{','.join(standardize_list)}]",
        f"model.dropout={dropout}",
        f"model.graph_regularization.graph_reg_lambda={graph_reg}",
        # The per-head `lambda` entries interpolate ${...graph_reg_lambda}, so overriding
        # the single scalar above retunes all nine regularized heads -- including to 0 for
        # the ablation, which is what makes lambda=0 a true no-graph-prior arm rather than
        # a mostly-off one.
        f"model.graph_regularization.graph_reg_layer={GRAPH_REG_LAYERS[reg_depth]}",
        f"regression_task.optimizer.lr={lr}",
        f"regression_task.optimizer.weight_decay={weight_decay}",
        f"data_module.num_workers={os.getenv('NUM_WORKERS', '4')}",
        # One W&B project per condition: expr | morph | expr_morph. The suffix marks the
        # round, so _007 runs never mix with _006 in the same project.
        f"wandb.project=torchcell_019_{CONDITION}_{PROJECT_SUFFIX}",
        "wandb.tags=[ws-run,ctrl-split,decoder,optuna,single-gpu,"
        f"{CONDITION},{decoder},{dist},trial-{trial.number},"
        f"graphreg-{'on' if graph_reg_on else 'off'},{reg_depth}]",
    ]

    with initialize_config_dir(version_base=None, config_dir=CONF_DIR):
        cfg = compose(config_name=BASE_CONFIG, overrides=overrides)

    print(
        f"[{CONDITION} w{WORKER_ID}] trial {trial.number}: heads={ACTIVE_HEADS} "
        f"emb={node_emb or '(none)'} learnable={learnable} "
        f"decoder={decoder} dist={dist} energy_rank={energy_rank} "
        f"graph_reg={graph_reg:g} ({'on' if graph_reg_on else 'ABLATION off'}, {reg_depth}) "
        f"lr={lr:.3g} wd={weight_decay:.3g} dropout={dropout} "
        f"norm+={normalize_list} std+={standardize_list}",
        flush=True,
    )
    print(OmegaConf.to_yaml(cfg.multitask), flush=True)

    metrics = run_training(cfg)
    torch.cuda.empty_cache()

    expr_r = metrics.get(EXPR_METRIC)
    morph_r = metrics.get(MORPH_METRIC)
    trial.set_user_attr("expr_pearson", expr_r)
    trial.set_user_attr("morph_pearson", morph_r)
    trial.set_user_attr("decoder", decoder)
    trial.set_user_attr("node_embeddings", node_emb or "(none)")
    trial.set_user_attr("learnable_embedding", learnable)
    trial.set_user_attr("dist", dist)
    # The mean-collapse diagnostic: per-instance stays high under collapse, so a large gap
    # vs per-feature is the signature. Recorded per trial so the sweep can be read for
    # "did the distributional loss actually prevent collapse?", not just the headline r.
    trial.set_user_attr(
        "morph_pearson_per_instance",
        metrics.get("val/morphology/pearson_per_instance_max"),
    )
    trial.set_user_attr(
        "expr_pearson_per_instance",
        metrics.get("val/expression/pearson_per_instance_max"),
    )
    trial.set_user_attr("energy_rank", energy_rank)
    trial.set_user_attr("graph_reg_on", graph_reg_on)
    trial.set_user_attr("graph_reg_depth", reg_depth)
    trial.set_user_attr("lr", lr)
    trial.set_user_attr("weight_decay", weight_decay)
    trial.set_user_attr("dropout", dropout)
    # THE SCORING-RULE VALUE ITSELF. For `dist=energy`, `val/loss` IS the energy score, and it
    # is the ONLY recorded quantity that can see what the joint covariance does.
    #
    # Why this matters, and why the round nearly missed it: the ranked objective is
    # `pearson_per_feature`, computed from `DistHead.point()` = mu. The energy head's global
    # factor V enters ONLY through Sigma = diag(sigma^2) + V V^T -- it does not touch mu. So
    # the `energy_rank in {0, 32}` ablation is STRUCTURALLY INVISIBLE to the ranked metric: a
    # null there is what the mathematics predicts, not evidence that modelling the joint fails.
    # Judging it needs a proper scoring rule, i.e. this value, compared within `dist=energy`
    # at matched rank.
    #
    # `_at_peak` (not `_max`) because a LOSS is minimized -- a max over epochs would report the
    # single worst epoch. Reading it at the ranked peak also keeps it paired with the point
    # estimate it accompanied. Recorded for every dist so cross-arm loss curves stay available,
    # but it is only COMPARABLE within a dist (different scoring rules are different units).
    trial.set_user_attr("val_loss", metrics.get("val/loss_at_peak"))
    trial.set_user_attr("val_loss_last", metrics.get("val/loss"))
    # CALIBRATION. Logged to W&B every val epoch by the training script; mirrored into the
    # trial so the sweep can be ranked on the point metric and READ on calibration in one
    # table. `_at_peak`, NOT `_max`: coverage has a TARGET (0.5 / 0.8) rather than a
    # direction, so its max over epochs is the most over-dispersed epoch -- the opposite of
    # the best one. `_at_peak` reads every metric at the epoch the run is actually ranked on,
    # which is also the only epoch where pairing calibration with the point estimate is
    # meaningful.
    for pheno in ("expression", "morphology"):
        for key in ("coverage_50", "coverage_80", "pit_ks"):
            value = metrics.get(f"val/{pheno}/calib/{key}_at_peak")
            if value is not None:
                trial.set_user_attr(f"{pheno}_{key}", value)

    if CONDITION == "expr_morph":
        # MULTI-OBJECTIVE: maximize BOTH honest metrics -> Optuna returns the Pareto front of
        # (expression, morphology). This is the scientific object: the configs on the frontier
        # of "good at both", NOT a hand-weighted scalar that hides the trade-off.
        if expr_r is None or morph_r is None:
            raise optuna.TrialPruned(
                f"expr_morph needs both metrics (expr={expr_r}, morph={morph_r})"
            )
        return expr_r, morph_r

    objective_metric = EXPR_METRIC if CONDITION == "expr" else MORPH_METRIC
    if objective_metric not in metrics:
        raise optuna.TrialPruned(
            f"{objective_metric} not logged (keys: {sorted(metrics)[:12]}...)"
        )
    return metrics[objective_metric]


def get_study() -> optuna.Study:
    """Create-or-load the study. expr_morph = MULTI-objective (maximize expr, maximize morph);
    expr/morph = single-objective.

    SAMPLER: Halton QMC, replacing TPE. TPE is an EXPLOIT-first sampler -- it fits a density
    over the good region and resamples from it -- which is right when the goal is the single
    best configuration and wrong when the goal is ATTRIBUTION. _007 is an attribution round:
    it exists to say which of lr / weight_decay / dropout carried `hp_profile`'s ~2x, and
    whether lambda=0 and rank=0 lose anything. That needs the axes COVERED, not the winner
    refined -- and a TPE run that concentrates in one corner leaves the ablation arms with
    too few samples to compare.

    Quasi-Monte-Carlo fills the unit cube by construction: a Halton sequence has low
    discrepancy, so every 1-D projection is near-uniform and every 2-D projection is
    well spread, at any prefix length. That is the property that makes marginal effects
    readable from a partial sweep -- and these workers get killed by the wall clock, so
    every sweep is a partial one.

    CAVEAT, verified against optuna 4.9.0: `QMCSampler` handles only FLOAT and INT
    distributions with its sequence; for a CategoricalDistribution it delegates to
    `independent_sampler` (a seeded random sampler). So the QMC guarantee applies to `lr` and
    `weight_decay` -- which is exactly why the optimizer axes were made continuous -- while
    the categorical axes (dist, energy_rank, graph_reg_on, node_embeddings, ...) are sampled
    independently at random. That is acceptable: they are small discrete sets that fill in
    quickly, and independent sampling keeps them uncorrelated with the continuous axes, which
    is what an attribution round wants.

    `seed=WORKER_ID` gives each of the 6 workers a different Halton scramble, so they cover
    complementary parts of the cube instead of replaying one sequence six times.
    """
    sampler = optuna.samplers.QMCSampler(
        qmc_type="halton",
        scramble=True,
        seed=WORKER_ID,
        independent_sampler=optuna.samplers.RandomSampler(seed=WORKER_ID),
        # The categorical fallback documented above is INTENDED, not a misconfiguration.
        # Left on, optuna emits one warning per categorical per trial -- 6 params x N trials
        # x 6 workers of identical text, which would bury the actual trial lines in the
        # slurm logs for three days.
        warn_independent_sampling=False,
    )
    common = dict(
        study_name=STUDY_NAME, storage=STORAGE, sampler=sampler, load_if_exists=True
    )
    if CONDITION == "expr_morph":
        return optuna.create_study(directions=["maximize", "maximize"], **common)
    return optuna.create_study(direction="maximize", **common)


def main() -> None:
    study = get_study()
    # --create-only: the slurm runs this ONCE (serialized) before the workers so they only
    # load_if_exists — avoids the fresh-DB DDL race. The directions logic lives HERE (one
    # place) so pre-create and workers always agree on single- vs multi-objective.
    if "--create-only" in sys.argv:
        print(f"[create-only] study={STUDY_NAME} directions={study.directions}", flush=True)
        return

    print(
        f"[{CONDITION} w{WORKER_ID}] study={STUDY_NAME} heads={ACTIVE_HEADS} "
        f"n_trials={N_TRIALS} multi_obj={CONDITION == 'expr_morph'} base={BASE_CONFIG}",
        flush=True,
    )
    study.optimize(objective, n_trials=N_TRIALS, catch=(Exception,))

    if WORKER_ID == 0:
        if CONDITION == "expr_morph":
            print(f"[expr_morph w0] Pareto front ({len(study.best_trials)} trials):")
            for t in study.best_trials[:10]:
                print(f"  t{t.number} (expr,morph)={[round(v, 4) for v in t.values]} {t.params}")
        else:
            print(f"[{CONDITION} w0] best={study.best_value:.4f} params={study.best_params}")


if __name__ == "__main__":
    import multiprocessing as mp

    mp.set_start_method("spawn", force=True)
    main()
