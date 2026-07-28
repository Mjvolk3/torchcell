# experiments/021-ozaydin-beta-carotene/scripts/optuna_metabolism_sweep.py
# [[experiments.021-ozaydin-beta-carotene.scripts.optuna_metabolism_sweep]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/021-ozaydin-beta-carotene/scripts/optuna_metabolism_sweep
"""OPTIMIZER x DISTRIBUTIONAL sweep driver (_004) for the three metabolism arms.

This is the metabolism port of the expression round `_007`
(`experiments/019-simb-multimodal/scripts/optuna_joint_sweep.py`): same decomposed optimizer
axes, same Halton QMC sampler, same distributional axis, same calibration bookkeeping. One
arm per GPU, three independent studies.

WHY A TWO-STAGE DESIGN, WHEN _007 IS SINGLE-STAGE
-------------------------------------------------
Because the metabolism objective is far noisier than the expression one, and by a mechanism
that is structural rather than incidental.

`pearson_per_feature` is a MEAN OVER FEATURES, so its run-to-run noise falls as
1/sqrt(F_eff). Expression has F = 6,127 and its trials are effectively denoised by the metric
itself; `betaxanthin` and `beta_carotene` have F = 1, so the objective is a SINGLE
correlation over ~473 validation strains. The `_002` studies re-proposed identical configs by
chance, which measured this directly -- same hyperparameters AND same seed:

    betaxanthin    trials  3/10/11   0.4216 / 0.4095 / 0.3584     sigma = 0.030
    betaxanthin    trials  19/31     0.4211 / 0.3896
    beta_carotene  trials  18/22     0.1993 / 0.1494              sigma = 0.025
    mulleder19     trials  7/10      0.1798 / 0.1780              (F=19, far quieter)

Against sigma = 0.030, betaxanthin's top FIVE `_002` configs span 0.021 -- 0.68 sigma, i.e.
statistically indistinguishable -- and the max over 36 trials carries a selection inflation
of about 2.0 sigma = 0.061, which puts the reported 0.4301 on a bias-corrected floor near
0.37. A single-stage sweep on a scalar arm can therefore rank AXES but cannot pick a WINNER,
and picking a winner is the point: the chosen betaxanthin model is what gets compared to
Merzbacher, and the chosen beta-carotene model is what gets applied to the CIT2
double-mutant panel.

    STAGE=screen   wide QMC search, one seed per trial. Answers "which axes matter?"
    STAGE=confirm  re-runs the top-K screen configs across R seeds and ranks on the MEAN.
                   Answers "which config is actually best?" -- sigma/sqrt(5) = 0.013 on
                   betaxanthin, which finally resolves the 0.02 gaps the screen cannot.

`cfg.seed` drives BOTH model init and the CellDataModule split (`random_seed=seed`), so the
confirm stage is repeated-random-subsampling validation, not merely an init re-roll. That is
the stronger estimator for "best in expectation over splits", which is the quantity we
actually want to select on.

WHAT IS INHERITED FROM _007 UNCHANGED
-------------------------------------
* lr and weight_decay CONTINUOUS log-uniform over the same ranges, dropout categorical.
  `_002` used the same unattributable `hp_profile` BUNDLE that _007 exists to decompose, and
  here TPE starved it further -- `baseline` got n=5 / n=4 / n=2 across the three arms while
  posting the BEST mean on two of them. The bundle has to go.
* Halton QMC with `seed=WORKER_ID`. Its low-discrepancy guarantee covers only the float axes
  (optuna delegates categoricals to the independent sampler); that is precisely why the
  optimizer axes are the continuous ones.
* Ranking on the PEAK (`_max`) of the phenotype-namespaced metric, computed from
  `DistHead.point()` so every distributional arm is compared on the same point estimate.
* Calibration (`calib/coverage_50`, `calib/coverage_80`, `calib/pit_ks`) recorded `_at_peak`
  and NOT ranked -- a model that predicts the marginal distribution is perfectly calibrated
  and useless.

WHAT HAD TO CHANGE, AND WHY (each is forced by a measurement, not a preference)
------------------------------------------------------------------------------
* `energy` and `energy_rank` are DROPPED on the two scalar arms. At F = 1 the predictive
  covariance Sigma = diag(sigma^2) + V V^T has V of shape [1, k], so V V^T is a scalar and
  the energy score degenerates to a Monte-Carlo estimate of CRPS -- strictly noisier than the
  closed form already in the sweep. Keeping it would be the phantom dimension this file's
  predecessor documents for `decoder`. `mulleder19` (F = 19) keeps it.
* `energy_rank` for mulleder19 is a GRID {0, 8}, not _007's measured 32. A 19x19 covariance
  has rank <= 19, so 32 cannot be realized; `DistHead` has no guard and would silently
  allocate V [19, 32]. The effective rank of the 19-AA residual correlation has not been
  measured -- `residual_covariance_diagnostic.py` re-run on this target is the follow-up --
  so it is SWEPT rather than asserted.
* `graph_reg_lambda` grids are PER ARM. Measured from the `_002` `graph_reg_ratio` attrs, the
  same nominal lambda buys very different prior strengths:
  ratio(6.5e-5) = 0.41 / 0.17 / 0.29 on betaxanthin / beta_carotene / mulleder19, i.e.
  parity (ratio = 1) at lambda = 1.6e-4 / 3.4e-4 / 2.2e-4. A shared lambda is NOT a shared
  prior, so each arm gets decades bracketing its own parity.
* `num_transformer_layers` stays SWEPT {2, 4} where _007 froze it at 4. _006 swept {2,4,6,8}
  on expression and found it inert, which is what licensed freezing there; `_002` measured a
  large, monotone, ARM-DEPENDENT effect here (betaxanthin 2 -> 0.288, 4 -> 0.197, 6 -> 0.124,
  while beta_carotene preferred 4). Freezing to expression's value would contradict our own
  measurement for at least one arm.
* `graph_reg_depth` is NOT swept. _007 can compare early [1] against middle [2] because L is
  frozen at 4; with L swept down to 2 (layers 0..1) a depth of [2] is out of range half the
  time, which would make the axis mean different things in different trials.
* `node_embeddings` is _007's FULL EIGHT, unchanged. An earlier cut of this round narrowed
  it to {prot_T5_all, random_1000} to buy resolution on the optimizer axes; that traded away
  the cross-round comparison, which is the point of mirroring _007. Restored.

ROUND _004 (this one) ADDS, on top of the above:
* THE PINNED CACHERA/MERZBACHER TEST SPLIT on the betaxanthin arm
  (`data_module.pinned_test_split_file`). Their 639 usable test ORFs are forced into OUR test
  split instead of being scattered by the 80/10/10 random split, which had left roughly 511
  of them in our TRAIN set -- so any score quoted against their Fig 4b would have been partly
  a training score. Ranking still happens on VAL, which is untouched and stays comparable.
* `ranked_smooth3`, the smoothed companion OF THE RANKED METRIC, derived from
  `OBJECTIVE_METRIC` rather than hardcoded to Pearson. `_003` recorded `pearson_smooth3` on
  every arm, which is the wrong metric on beta-carotene (it ranks on Spearman).

Environment (set by the slurm launcher):
    ARM                 betaxanthin | beta_carotene | mulleder19   (required)
    STAGE               screen | confirm                           (default: screen)
    OPTUNA_STORAGE      sqlite:////<scratch>/.../optuna_020_<arm>.db   (required)
    OPTUNA_STUDY_NAME   default: <arm>_004_<stage>
    OPTUNA_N_TRIALS     trials THIS worker runs (default: 60 screen)
    OPTUNA_WORKER_ID    seeds the Halton scramble (default: 0)
    METAB_BASE_CONFIG   base Hydra config (default: <arm>_004)
    CONFIRM_TOP_K       configs promoted from screen to confirm (default: 8)
    NUM_WORKERS         dataloader workers per trial (default: 4)
"""

import os
import os.path as osp
import statistics as st
import sys
from typing import Any

import optuna
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

_REPO_ROOT = osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))
_TRAINER_DIR = osp.join(_REPO_ROOT, "experiments", "019-simb-multimodal", "scripts")
if _TRAINER_DIR not in sys.path:
    sys.path.insert(0, _TRAINER_DIR)

from train_cgt_multitask import run_training  # type: ignore[import-not-found]  # noqa: E402

CONF_DIR = osp.abspath(osp.join(osp.dirname(__file__), "../conf"))
ARM = os.environ["ARM"]
STAGE = os.getenv("STAGE", "screen")
BASE_CONFIG = os.getenv("METAB_BASE_CONFIG", f"{ARM}_004")
STORAGE = os.environ["OPTUNA_STORAGE"]
STUDY_NAME = os.getenv("OPTUNA_STUDY_NAME", f"{ARM}_004_{STAGE}")
N_TRIALS = int(os.getenv("OPTUNA_N_TRIALS", "60"))
WORKER_ID = int(os.getenv("OPTUNA_WORKER_ID", "0"))
CONFIRM_TOP_K = int(os.getenv("CONFIRM_TOP_K", "8"))
#: Completed screen trials after which the launcher switches this arm to `confirm`.
SCREEN_TARGET = int(os.getenv("SCREEN_TARGET", "60"))
SCREEN_STUDY_NAME = os.getenv("SCREEN_STUDY_NAME", f"{ARM}_004_screen")

assert STAGE in ("screen", "confirm"), f"STAGE must be screen|confirm, got {STAGE!r}"

#: Ranking metric per arm. Beta-carotene is ranked on SPEARMAN and that is not a preference:
#: Ozaydin's readout is a SUBJECTIVE ORDINAL on -5..+5, so a Pearson on it asserts an interval
#: scale the measurement does not have, and would select models that fit a scale artifact.
OBJECTIVE_METRIC = {
    "betaxanthin": "val/betaxanthin/pearson_per_feature_max",
    "beta_carotene": "val/beta_carotene/spearman_per_feature_max",
    "mulleder19": "val/mulleder19/pearson_per_feature_max",
}[ARM]

#: The de-biased companion OF THE RANKED METRIC -- derived from it rather than hardcoded.
#: An earlier cut recorded `pearson_smooth3` on every arm, which is the wrong metric on
#: beta-carotene: that arm RANKS ON SPEARMAN, so its spikiness check was comparing a Spearman
#: maximum against a Pearson rolling mean. The tell was arithmetically impossible values
#: (`smooth3 > max`, which cannot happen within one metric, since the max of a 3-epoch mean
#: is bounded by the max of the series). Deriving the key removes the chance of them drifting
#: apart again.
RANKED_SMOOTH3 = OBJECTIVE_METRIC.replace("_max", "_smooth3_max")

#: Screen seed. 42 rather than _007's 0, deliberately: the `_002` studies ran at 42, so a
#: screen winner can be read against 36-40 trials of existing history at the same seed.
SCREEN_SEED = 42
#: Confirm seeds. R = 5 takes betaxanthin's sigma = 0.030 down to 0.013, which is what makes
#: the ~0.02 gaps between its top configs readable. 42 leads so the screen run is reused as
#: one of the five rather than repeated.
CONFIRM_SEEDS = [42, 0, 1, 2, 3]

#: NO DECODER AXIS -- verified, not assumed. `multitask.decoder` switches the structural form
#: of the inherited `global` (morphology) head only. All three metabolism targets are read by
#: the metabolism subclass's own `ProductScalarHead` / `MetabolomeVectorHead`, which are plain
#: MLPs and never consult it.
DECODER = "s1_pool"

#: The probabilistic axis -- what this round spends its budget on. `point` (MSE) stays dropped:
#: its optimum IS the conditional mean, i.e. the mean-collapse the honest metric punishes, and
#: `_002` already measured it last on all three arms (betaxanthin 0.172 vs crps 0.275).
#: `energy` appears only where a joint distribution exists to model -- see module docstring.
DIST_CHOICES = {
    "betaxanthin": ["crps", "quantile", "laplace_crps", "nll"],
    "beta_carotene": ["crps", "quantile", "laplace_crps", "nll"],
    "mulleder19": ["crps", "quantile", "laplace_crps", "nll", "energy"],
}[ARM]

#: Rank of the low-rank factor V in Sigma = diag(sigma^2) + V V^T, mulleder19 only. Swept, not
#: asserted: 0 is the diagonal-only ablation that isolates "does modelling the joint pay?" from
#: "is energy a better objective?", and 8 is a real factor that fits inside F = 19.
ENERGY_RANKS = [0, 8]

#: MEASURED per arm from the `_002` `graph_reg_ratio` user attrs (graph term / data term),
#: as decades bracketing that arm's own parity. Reading these as interchangeable across arms
#: is exactly the error the shared 6.5e-5 made.
GRAPH_REG_LAMBDAS = {
    "betaxanthin": [1.6e-6, 1.6e-5, 1.6e-4, 1.6e-3],
    "beta_carotene": [3.4e-6, 3.4e-5, 3.4e-4, 3.4e-3],
    "mulleder19": [2.2e-6, 2.2e-5, 2.2e-4, 2.2e-3],
}[ARM]

#: THE FULL _007 EMBEDDING AXIS, verbatim. An earlier cut of this round narrowed it to
#: {prot_T5_all, random_1000} to buy resolution on the optimizer axes; that traded away the
#: comparison with the expression round, which is the reason to mirror _007 in the first
#: place. All eight are coverage-audited (6607 genes, 0 missing, 0 zero-vectors) -- variants
#: that fail that bar are deliberately absent (one_hot_gene / fudt_downstream-alone miss the
#: 28 mitochondrial Q0* genes; the esm2/prot_T5 `no_dubious` / `no_uncharacterized` variants
#: emit 684/668 ALL-ZERO vectors, i.e. silently information-free for the hardest ORFs).
#:
#:   codon_frequency (64d)          the only option narrower than hidden, so no compression
#:   calm (768d), fudt_upstream (768d), prot_T5_all (1024d),
#:   esm2_t33_650M_UR50D_all (1280d), nt_window_5979 (2560d)   real sequence/protein content
#:   fudt_upstream,fudt_downstream (1536d)  promoter + terminator, i.e. the REGULATORY axis,
#:     orthogonal to the protein-coding one; a comma-joined entry becomes a multi-embedding
#:     override that the model concatenates
#:   random_1000 (1000d)            CONTROL -- unique but meaningless, and WIDTH-MATCHED.
#:     _007's upgrade from random_100, which at 7-25x narrower than the real embeddings made
#:     "content wins" partly a statement about dimensionality.
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

TARGET_NORMS = ["zscore", "yeo_johnson"]


def _suggest_params(trial: optuna.Trial) -> dict[str, Any]:
    """Sample one configuration.

    Shared by both stages so an enqueued confirm trial replays a screen trial's parameters
    through the identical code path.
    """
    params = {
        "node_embeddings": trial.suggest_categorical("node_embeddings", NODE_EMBEDDINGS),
        "dist": trial.suggest_categorical("dist", DIST_CHOICES),
        "num_transformer_layers": trial.suggest_categorical(
            "num_transformer_layers", [2, 4]
        ),
        # THE ABLATION. lambda = 0 exactly is the only honest test of whether the graph prior
        # carries the arm. `_002` suggests it does for betaxanthin (0.113 at lambda=0 vs 0.280
        # at 6.5e-5) but NOT for beta-carotene, where lambda=0 scored highest -- on n=5 and
        # n=17 respectively, under a TPE sampler that allocated them unevenly. Re-asked here
        # under QMC, where the level counts are balanced by construction.
        "graph_reg_on": trial.suggest_categorical("graph_reg_on", [False, True]),
        "graph_reg_lambda": trial.suggest_categorical(
            "graph_reg_lambda", GRAPH_REG_LAMBDAS
        ),
        # The hp_profile bundle, DECOMPOSED. Continuous log-uniform over ranges that CONTAIN
        # all three `_002` profiles rather than bracketing a guess: baseline lr 3e-4,
        # aggressive 1e-3, and 010's 1e-4 all sit inside; weight_decay spans aggressive's 1e-4
        # and 010's 1e-8 and reaches low enough to be effectively off.
        "lr": trial.suggest_float("lr", 5e-5, 2e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-9, 1e-3, log=True),
        # Categorical: dropout is a coarse dial and the question is "any vs none", not the
        # third decimal.
        "dropout": trial.suggest_categorical("dropout", [0.0, 0.1, 0.2]),
        "target_norm": trial.suggest_categorical("target_norm", TARGET_NORMS),
    }
    # Suggested for EVERY trial on the arm that has it, not only the energy ones: optuna
    # defines a search space by which params a trial calls, so a conditional suggest would
    # make the space ragged -- and QMC assumes a fixed-dimension cube. Inert unless
    # dist == energy, and recorded either way so a filter never hits a null.
    if "energy" in DIST_CHOICES:
        params["energy_rank"] = trial.suggest_categorical("energy_rank", ENERGY_RANKS)
    return params


def _overrides(params: dict[str, Any], seed: int) -> list[str]:
    """Translate a sampled configuration into Hydra overrides."""
    graph_reg = params["graph_reg_lambda"] if params["graph_reg_on"] else 0.0
    normalize_list = [] if params["target_norm"] == "zscore" else [ARM]
    standardize_list = [ARM] if params["target_norm"] == "zscore" else []
    ov = [
        f"seed={seed}",
        f"multitask.active_heads=[{ARM}]",
        f"cell_dataset.node_embeddings=[{params['node_embeddings']}]",
        "model.learnable_embedding.enabled=false",
        f"multitask.decoder={DECODER}",
        f"multitask.dist={params['dist']}",
        f"multitask.normalize_vector_targets=[{','.join(normalize_list)}]",
        f"multitask.standardize_per_feature_target=[{','.join(standardize_list)}]",
        f"model.num_transformer_layers={params['num_transformer_layers']}",
        f"model.dropout={params['dropout']}",
        # The nine per-head `lambda` entries interpolate this single scalar, so overriding it
        # retunes all nine regularized heads at once -- including to 0, which is what makes the
        # ablation a true no-prior arm rather than a mostly-off one.
        f"model.graph_regularization.graph_reg_lambda={graph_reg}",
        f"regression_task.optimizer.lr={params['lr']}",
        f"regression_task.optimizer.weight_decay={params['weight_decay']}",
        f"data_module.num_workers={os.getenv('NUM_WORKERS', '4')}",
        f"wandb.project=torchcell_020_{ARM}_v3",
        f"wandb.tags=[020,optuna,{STAGE},{ARM},{params['node_embeddings']},"
        f"{params['dist']},seed-{seed},graphreg-{'on' if params['graph_reg_on'] else 'off'}]",
    ]
    if "energy_rank" in params:
        ov.append(f"multitask.energy_rank={params['energy_rank']}")
    return ov


def _train_once(params: dict[str, Any], seed: int, label: str) -> dict[str, float]:
    """Compose the config for one (config, seed) pair and run it."""
    with initialize_config_dir(version_base=None, config_dir=CONF_DIR):
        cfg = compose(config_name=BASE_CONFIG, overrides=_overrides(params, seed))
    print(
        f"[{ARM} {STAGE} w{WORKER_ID}] {label} seed={seed}: "
        f"emb={params['node_embeddings']} dist={params['dist']} "
        f"L={params['num_transformer_layers']} "
        f"graph_reg={(params['graph_reg_lambda'] if params['graph_reg_on'] else 0.0):g} "
        f"({'on' if params['graph_reg_on'] else 'ABLATION off'}) "
        f"lr={params['lr']:.3g} wd={params['weight_decay']:.3g} "
        f"dropout={params['dropout']} norm={params['target_norm']}"
        + (f" energy_rank={params['energy_rank']}" if "energy_rank" in params else ""),
        flush=True,
    )
    print(OmegaConf.to_yaml(cfg.multitask), flush=True)
    metrics: dict[str, float] = run_training(cfg)
    torch.cuda.empty_cache()
    return metrics


def _record(trial: optuna.Trial, metrics: dict[str, float], suffix: str = "") -> None:
    """Mirror the diagnostic metrics into the trial so the study reads standalone.

    Recorded, never ranked. `pred_sd_ratio` is the mean-collapse diagnostic for a SCALAR head
    (`pearson_per_instance` is undefined at F = 1 and came back None on two of three arms in
    `_002`); `n_val_epochs` and `peak_epoch` are the covariates for the max-over-epochs bias,
    which on `_002` mulleder19 correlated with the objective at r = +0.75.
    """
    for name, key in (
        ("pearson", f"val/{ARM}/pearson_per_feature_max"),
        ("spearman", f"val/{ARM}/spearman_per_feature_max"),
        # `ranked_smooth3` is the one to read: it is the smoothed companion of whichever
        # metric THIS arm is ranked on. Both raw variants are kept alongside so a study can
        # be re-read on the other correlation without re-running.
        ("ranked_smooth3", RANKED_SMOOTH3),
        ("pearson_smooth3", f"val/{ARM}/pearson_per_feature_smooth3_max"),
        ("spearman_smooth3", f"val/{ARM}/spearman_per_feature_smooth3_max"),
        ("pred_sd_ratio", f"val/{ARM}/pred_sd_ratio_at_peak"),
        ("graph_reg_ratio", "val/graph_reg/ratio_to_data_at_peak"),
        ("n_val_epochs", "val/n_val_epochs"),
        ("peak_epoch", "val/peak_epoch"),
    ):
        trial.set_user_attr(f"{name}{suffix}", metrics.get(key))
    for key in ("coverage_50", "coverage_80", "pit_ks"):
        trial.set_user_attr(
            f"calib_{key}{suffix}", metrics.get(f"val/{ARM}/calib/{key}_at_peak")
        )


def objective_screen(trial: optuna.Trial) -> float:
    """One config, one seed. Wide coverage for attribution."""
    params = _suggest_params(trial)
    metrics = _train_once(params, SCREEN_SEED, f"trial {trial.number}")
    _record(trial, metrics)
    trial.set_user_attr("seed", SCREEN_SEED)
    if OBJECTIVE_METRIC not in metrics:
        raise optuna.TrialPruned(
            f"{OBJECTIVE_METRIC} not logged (keys: {sorted(metrics)[:12]}...)"
        )
    return float(metrics[OBJECTIVE_METRIC])


def objective_confirm(trial: optuna.Trial) -> float:
    """One config, R seeds, ranked on the MEAN.

    The per-seed values and their standard error are recorded so the report can state a
    difference WITH its uncertainty instead of a bare maximum. A confirm winner whose SE
    overlaps the runner-up is a tie, and saying so is the whole point of the stage.
    """
    params = _suggest_params(trial)
    values: list[float] = []
    for i, seed in enumerate(CONFIRM_SEEDS):
        metrics = _train_once(params, seed, f"trial {trial.number} rep {i + 1}")
        _record(trial, metrics, suffix=f"_s{seed}")
        if OBJECTIVE_METRIC not in metrics:
            raise optuna.TrialPruned(f"{OBJECTIVE_METRIC} not logged at seed {seed}")
        values.append(float(metrics[OBJECTIVE_METRIC]))
        # Written every repeat so a walltime kill mid-config still leaves the partial
        # evidence in the study rather than discarding the seeds already paid for.
        trial.set_user_attr("values_so_far", list(values))
    mean = st.mean(values)
    trial.set_user_attr("seed_values", values)
    trial.set_user_attr("seed_mean", mean)
    trial.set_user_attr("seed_sd", st.stdev(values) if len(values) > 1 else None)
    trial.set_user_attr(
        "seed_sem", st.stdev(values) / len(values) ** 0.5 if len(values) > 1 else None
    )
    print(
        f"[{ARM} confirm] trial {trial.number}: mean={mean:.4f} "
        f"values={[round(v, 4) for v in values]}",
        flush=True,
    )
    return mean


def get_study() -> optuna.Study:
    """Create-or-load. Single-objective maximize -- each arm has exactly one honest metric.

    SAMPLER, screen stage: Halton QMC rather than TPE. TPE is EXPLOIT-first -- it fits a
    density over the good region and resamples it -- which is what produced `_002`'s
    unreadable marginals: `hp_profile=baseline` was sampled n=2 on mulleder19 while posting
    the best mean, and `graph_reg_lambda=0` n=2 while posting the worst. Attribution needs the
    levels COVERED, not the incumbent refined. QMC fills the cube by low discrepancy, and
    (per optuna 4.9.0) delegates categoricals to a seeded random sampler -- which balances the
    level counts, which is what `_002` lacked.

    Confirm stage runs enqueued configurations only, so its sampler is never consulted for the
    trials that matter; it is constructed identically for consistency.
    """
    sampler = optuna.samplers.QMCSampler(
        qmc_type="halton",
        scramble=True,
        seed=WORKER_ID,
        independent_sampler=optuna.samplers.RandomSampler(seed=WORKER_ID),
        # The categorical fallback is INTENDED. Left on, optuna emits one warning per
        # categorical per trial, which would bury the trial lines in a multi-day slurm log.
        warn_independent_sampling=False,
    )
    return optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE,
        sampler=sampler,
        load_if_exists=True,
        direction="maximize",
    )


def _param_key(trial: optuna.trial.FrozenTrial) -> tuple[tuple[str, Any], ...]:
    """Hashable identity of a trial's configuration.

    An ENQUEUED (WAITING) trial has an EMPTY ``.params`` -- optuna keeps its values in
    ``system_attrs["fixed_params"]`` until the trial actually runs and re-suggests them. Using
    ``.params`` alone therefore silently fails to recognize already-enqueued configs, and a
    requeued confirm job would enqueue the whole top-K a second time.
    """
    params = trial.params or trial.system_attrs.get("fixed_params", {})
    return tuple(sorted(params.items()))


def _promote_from_screen(study: optuna.Study) -> int:
    """Enqueue the top-K screen configurations into the confirm study.

    Deduplicated on the parameter dict: QMC re-proposes points, and replicating the same
    configuration twice would spend a fifth of the confirm budget re-measuring one number.
    Idempotent against an already-populated confirm study, so a requeue after a walltime kill
    resumes rather than double-enqueues.
    """
    screen = optuna.load_study(study_name=SCREEN_STUDY_NAME, storage=STORAGE)
    done = [t for t in screen.trials if t.state.name == "COMPLETE" and t.values]
    done.sort(key=lambda t: t.values[0], reverse=True)
    already = {
        _param_key(t)
        for t in study.trials
        # FAIL is deliberately absent: a config whose confirm run died should be re-promotable.
        if t.state.name in ("COMPLETE", "RUNNING", "WAITING")
    }
    # CONFIRM_TOP_K caps the study TOTAL, not this call: a requeued job must top the confirm
    # set back up to K after failures, not append another K on top of what already ran.
    n_present = len(already)
    budget = CONFIRM_TOP_K - n_present
    promoted = 0
    for t in done:
        if promoted >= budget:
            break
        key = _param_key(t)
        if key in already:
            continue
        already.add(key)
        study.enqueue_trial(t.params)
        promoted += 1
        print(
            f"[{ARM} confirm] promoted screen t{t.number} "
            f"({t.values[0]:.4f}) {t.params}",
            flush=True,
        )
    print(
        f"[{ARM} confirm] {promoted} configs enqueued (budget {budget}, "
        f"{n_present} already present) from {len(done)} completed screen trials "
        f"in study {SCREEN_STUDY_NAME!r}",
        flush=True,
    )
    return promoted


def _n_complete(study_name: str) -> int:
    """Completed-trial count for a study that may not exist yet."""
    try:
        study = optuna.load_study(study_name=study_name, storage=STORAGE)
    except KeyError:
        return 0
    return len([t for t in study.trials if t.state.name == "COMPLETE" and t.values])


def _resolve_stage() -> None:
    """Print, as the LAST line of stdout, the stage this job should run: screen|confirm|done.

    The launcher resubmits itself until a wall-clock deadline, so the stage cannot be baked
    into the slurm script -- a requeue has to pick up where the previous job stopped. Deciding
    from COMPLETED trial counts makes that idempotent, and degrades correctly when a job is
    killed mid-trial, because an unfinished trial is not COMPLETE and so is not counted.

    `done` matters as much as the other two: once the confirm study holds CONFIRM_TOP_K
    completed trials there is no work left, and without this the deadline-guarded chain would
    keep resubmitting a job that promotes nothing -- holding a GPU for hours to do nothing.
    """
    if _n_complete(SCREEN_STUDY_NAME) < SCREEN_TARGET:
        print("screen")
        return
    print("done" if _n_complete(f"{ARM}_004_confirm") >= CONFIRM_TOP_K else "confirm")


def main() -> None:
    # --resolve-stage runs BEFORE the study for this stage exists, so it must not build one.
    if "--resolve-stage" in sys.argv:
        _resolve_stage()
        return

    study = get_study()
    # --create-only: slurm runs this ONCE before the worker so a fresh SQLite file's DDL is
    # not raced -- each worker spends >15s importing torch before it would reach create_study.
    if "--create-only" in sys.argv:
        print(
            f"[create-only] study={STUDY_NAME} direction={study.direction}", flush=True
        )
        return

    if STAGE == "confirm":
        n = _promote_from_screen(study)
        if n == 0:
            print(f"[{ARM} confirm] nothing to promote -- exiting.", flush=True)
            return
        study.optimize(objective_confirm, n_trials=n, catch=(Exception,))
    else:
        print(
            f"[{ARM} screen w{WORKER_ID}] study={STUDY_NAME} "
            f"objective={OBJECTIVE_METRIC} n_trials={N_TRIALS} base={BASE_CONFIG} "
            f"dist={DIST_CHOICES}",
            flush=True,
        )
        study.optimize(objective_screen, n_trials=N_TRIALS, catch=(Exception,))

    complete = [t for t in study.trials if t.state.name == "COMPLETE" and t.values]
    if WORKER_ID == 0 and complete:
        best = max(complete, key=lambda t: t.values[0])
        sem = best.user_attrs.get("seed_sem")
        sem_txt = f" +- {sem:.4f} (SEM over {len(CONFIRM_SEEDS)} seeds)" if sem else ""
        print(
            f"[{ARM} {STAGE} w0] best={best.values[0]:.4f}{sem_txt} params={best.params}"
        )


if __name__ == "__main__":
    import multiprocessing as mp

    mp.set_start_method("spawn", force=True)
    main()
