# experiments/019-simb-multimodal/scripts/metabolism_grid_runner.py
# [[experiments.019-simb-multimodal.scripts.metabolism_grid_runner]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/metabolism_grid_runner
"""EIGHT SETTINGS, EACH TRAINED FOR THE WHOLE ALLOCATION -- the Delta metabolism runs.

This REPLACES the Optuna search (`optuna_metabolism_sweep.py`) for the Delta jobs. Two
measurements force the shape, and the second one is the important one.

(1) MORE SEARCH CANNOT HELP -- the arms are noisier than the gaps being ranked
-----------------------------------------------------------------------------
Replicate noise on `pearson_per_feature` is sigma = 0.030 (betaxanthin) and 0.025
(beta-carotene), measured directly from `_002` trials that re-proposed identical
hyperparameters AND an identical seed:

    betaxanthin    trials  3/10/11   0.4216 / 0.4095 / 0.3584     sigma = 0.030
    beta_carotene  trials 18/22      0.1993 / 0.1494              sigma = 0.025

`_002`'s top FIVE betaxanthin configs spanned 0.021 -- 0.68 sigma. A single-seed search can
rank AXES; it cannot pick a WINNER, and a third one would buy more unreadable trials. So the
budget goes to eight FIXED settings instead of a search.

(2) EVERY METABOLISM RUN SO FAR MAY HAVE BEEN SCORED IN THE DIP
---------------------------------------------------------------
This is the reason the runs are LONG rather than merely replicated. From the 019 expression
wave-6 measurements ([[scratch.2026.07.30.132219-exp-design]]), on the same model and the
same trainer:

    * Smoothed val Pearson peaks ~0.14 at epoch 85-136, FALLS to 0.08-0.11 by epoch 200-300,
      then RISES to a project-best 0.1980 at epoch 1367.
      "Every arm ever scored at 300-400 epochs was scored in the dip."
    * Nothing had converged at 1,500 epochs (eval-mode train Pearson still climbing,
      max == last in 4 of 4 runs).
    * val LOSS and val PEARSON move in OPPOSITE directions -- loss bottoms at epoch 103-136
      and then rises while Pearson keeps climbing. So "the loss stopped improving" is not
      evidence that the metric has.

Now compare the metabolism history: every `_002`/`_003`/`_004` run early-stopped on a
patience of 40-50 and finished at a median of ~71-135 epochs. If metabolism has the same
dip-then-climb shape, ALL of those runs stopped at or before the first bump -- and the 0.4301
we have been quoting is not this model's ceiling, it is its first local maximum.

That is a hypothesis, not a fact, because it has never been tested on these targets. Testing
it is cheap and is exactly what 48 h on four A40s is for: run the eight settings LONG and
look at the shape of the curve.

    max_epochs 10,000 (a ceiling, not a target) + early stopping OFF
    + `trainer.max_time_s`, a WALL-CLOCK budget handed to a Lightning `Timer`.

The time budget is what makes this runnable. It stops training gracefully, so `fit` returns
and the metric snapshot, the test pass and the prediction dump all still happen -- where a
slurm kill mid-`fit` would lose all three and leave the trial RUNNING rather than COMPLETE.
It also spends exactly the compute available instead of an epoch cap guessed in advance.

THE 8 RUNS: FOUR SETTINGS x TWO INIT SEEDS
-------------------------------------------
A 2x2 factorial, fully crossed, replicated twice. Replication is affordable here only because
the SPLIT IS NOW PINNED (`data_module.split_seed: 0`), so `seed` varies weight initialization
alone. 019 measured what that separation buys: with one knob driving both, between-seed sd was
0.0444 against a within-seed across-arm sd of 0.0058 -- the nuisance axis was 7.7x the signal
axis, and at n=4 the minimum detectable effect was 7.9x the effect being chased. Two runs of
the same setting are now a paired comparison rather than two draws from a wide nuisance
distribution.

WHAT IT COSTS, stated plainly: the absolute level belongs to ONE validation draw. Arm
RANKINGS transfer; the number does not. If a generalizable absolute level is wanted later,
the fix is K-fold with arms paired on fold, not a different single seed.

WHAT ROUND 0 STILL DOES NOT ANSWER: a 0.02 gap between settings. At two seeds the SE is
sigma/sqrt(2) = 0.021. The primary artifact of this round is the CURVE -- where the target
saturates -- not a leaderboard.

RUNTIME MODEL
-------------
Work is enqueued ROUND-MAJOR -- round r contributes one seed to EACH setting -- so a
wall-clock kill leaves every setting with the same number of seeds and the factorial stays
balanced. Seed-major would leave 12 seeds on two settings and none on six.

Every worker computes its trial's `max_time_s` from the job deadline when it CLAIMS the
trial, so the eight round-0 runs all end together just inside the wall clock, and a
replicate started later simply gets whatever is left.

Environment (set by the slurm launcher):
    ARM                    betaxanthin | mulleder19 | beta_carotene | bx_ctrl | bx_m19
    GRID_CONF_DIR          absolute path to the arm's Hydra conf dir      (required)
    GRID_BASE_CONFIG       Hydra base config name                        (required)
    OPTUNA_STORAGE         sqlite:///.../optuna_<exp>_<arm>_grid.db       (required)
    OPTUNA_STUDY_NAME      default: <arm>_grid_000
    GRID_ROUNDS            seed rounds to enqueue (default 3)
    GRID_DEADLINE_EPOCH    unix seconds -- the job's wall clock (required for long runs)
    GRID_MIN_TRIAL_S       don't claim work with less than this left (default 3600)
    GRID_TEARDOWN_S        reserve after training for test + dump (default 900)
    OPTUNA_WORKER_ID       0..N-1, for logging only
    NUM_WORKERS            dataloader workers per trial
    GRID_MANIFEST_DIR      where --create-only writes grid_manifest_<arm>.json (optional)

Modes:
    --create-only   create the study, enqueue every (setting, seed), write the manifest.
                    Run ONCE from the launcher before the workers, so a fresh SQLite file's
                    DDL is not raced by processes that each spend >15s importing torch.
    (no flag)       run as a worker: drain the queue until it is empty or the deadline hits.
"""

import itertools
import json
import os
import os.path as osp
import sys
import time
from typing import Any

import optuna
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

_REPO_ROOT = osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))
_TRAINER_DIR = osp.join(_REPO_ROOT, "experiments", "019-simb-multimodal", "scripts")
if _TRAINER_DIR not in sys.path:
    sys.path.insert(0, _TRAINER_DIR)

from train_cgt_multitask import run_training  # type: ignore[import-not-found]  # noqa: E402, I001

ARM = os.environ["ARM"]
CONF_DIR = os.environ["GRID_CONF_DIR"]
BASE_CONFIG = os.environ["GRID_BASE_CONFIG"]
STORAGE = os.environ["OPTUNA_STORAGE"]
STUDY_NAME = os.getenv("OPTUNA_STUDY_NAME", f"{ARM}_grid_000")
ROUNDS = int(os.getenv("GRID_ROUNDS", "3"))
WORKER_ID = int(os.getenv("OPTUNA_WORKER_ID", "0"))
DEADLINE = float(os.getenv("GRID_DEADLINE_EPOCH", "0")) or None
#: Don't claim a trial with less than this much wall clock left -- an hour of training is
#: not enough to say anything about where a curve saturates, and the slot is better left idle
#: than filled with a run nobody can read.
MIN_TRIAL_S = float(os.getenv("GRID_MIN_TRIAL_S", "3600"))
#: Reserved AFTER training for the test pass, the prediction dump and wandb teardown. Taken
#: out of the trial's time budget rather than hoped for: the whole point of the graceful
#: `Timer` stop is that those three still happen.
TEARDOWN_S = float(os.getenv("GRID_TEARDOWN_S", "900"))

#: Seeds, in the order rounds consume them. These are now INIT-ONLY: the configs pin
#: `data_module.split_seed: 0`, so `cfg.seed` no longer selects the partition. Round 0 runs
#: every setting at 42 (the seed the whole `_002`/`_003`/`_004` history used), round 1
#: replicates at 0, and any further round is spare capacity for a setting that finished early.
SEEDS = [42, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

#: Heads active per arm. The two `bx_*` arms are the CONTROLLED auxiliary-task pair: same
#: restricted instance set (`require_head_targets` in their config), same settings, same
#: seeds -- differing ONLY in whether the metabolome head is attached. `bx_m19 - bx_ctrl`
#: read within a (setting, seed) cell is then a PAIRED difference, which removes the split
#: and init variance that dominates an unpaired comparison at this sample size.
ACTIVE_HEADS = {
    "betaxanthin": ["betaxanthin"],
    "beta_carotene": ["beta_carotene"],
    "mulleder19": ["mulleder19"],
    "bx_ctrl": ["betaxanthin"],
    "bx_m19": ["betaxanthin", "mulleder19"],
}[ARM]

#: The metric each arm is RANKED on. Beta-carotene is SPEARMAN and that is not a preference:
#: Ozaydin's readout is a subjective ordinal on -5..+5, so a Pearson asserts an interval
#: scale the measurement does not have. Both `bx_*` arms rank on BETAXANTHIN -- the metabolome
#: is an auxiliary task there, not a second objective.
OBJECTIVE_METRIC = {
    "betaxanthin": "val/betaxanthin/pearson_per_feature_max",
    "beta_carotene": "val/beta_carotene/spearman_per_feature_max",
    "mulleder19": "val/mulleder19/pearson_per_feature_max",
    "bx_ctrl": "val/betaxanthin/pearson_per_feature_max",
    "bx_m19": "val/betaxanthin/pearson_per_feature_max",
}[ARM]
RANK_PHENO = OBJECTIVE_METRIC.split("/")[1]
RANKED_SMOOTH3 = OBJECTIVE_METRIC.replace("_max", "_smooth3_max")

#: WHAT EVERY SETTING ON AN ARM HOLDS FIXED. Each entry is a measurement, not a default.
#:
#: `node_embeddings` -- prot_T5_all on all four arms. It led the kNN embedding probe with zero
#:   learned parameters and won the `_002` studies on betaxanthin and mulleder19. The `_003`/
#:   `_004` screens promoted `calm` / `random_1000` / `codon_frequency` on 17-19 trials at
#:   sigma 0.030, i.e. inside noise -- and a CONTROL embedding topping a 17-trial screen is
#:   evidence about the screen, not about the embedding.
#: `learnable_embedding` -- false, and structurally forced: every strain is a single deletion,
#:   so a strain-level split IS a gene-level split and a val gene's free row never receives a
#:   gradient (observed: train r 0.86 / val r 0.02).
#: `perturbation_head.num_heads` -- 6, the value every run up to and including `_003` used.
#:   `_004` rebound it to 9 IN THE SAME ROUND as it introduced the pinned split, and that
#:   round's best fell to 0.2469 from `_003`'s 0.4050. Two changes, one drop, no attribution.
#:   The pinned split is REQUIRED by the Merzbacher question, so the other change is the one
#:   that gets reverted; re-testing 9 heads is a follow-up, not a factor of this grid.
#: `dist` -- FROZEN. 019's `_007` settled it at n ~ 60 per mode: the five distributional
#:   modes are within noise of each other on accuracy (means 0.017-0.028 against within-mode
#:   sd 0.030-0.040) while calibration succeeded at both levels. "The distributional axis is
#:   not the lever." Each arm takes its own measured best. `point` (MSE) was never a
#:   candidate: its optimum IS the conditional mean, i.e. the mean-collapse the metric
#:   punishes.
#: `target_norm` -- FROZEN at zscore. `_002` preferred yeo_johnson and `_003` preferred
#:   zscore, so it is unsettled -- but both readings came from runs that stopped near epoch
#:   100, and this round exists because such readings are suspect. zscore is also the
#:   anti-mean-collapse lever and what the joint arm's metabolome head needs.
#: `dropout` -- FROZEN at 0.1, the level 019 measured paired against 0.0 (+0.0145, +0.0565).
#:   Higher levels are not swept here.
#: `graph_reg_lambda` -- GONE, not frozen: there is no KL term any more. `attention_mask`
#:   burns all nine relations into attention structurally and `_010` set lambda to 0. Keeping
#:   a lambda sweep would apply a hard constraint and a soft penalty to the same nine heads
#:   AND give back the 1.5-1.7x per-epoch speedup the mask buys (the KL needs attention
#:   WEIGHTS, so it must materialize a [1, 9, 6608, 6608] matmul and cannot use the fused
#:   kernel). At 48 h that speedup IS the experiment.
#: `num_transformer_layers` -- FROZEN at 6, matching `_012`. Metabolism `_002` measured a
#:   preference for L=2, but that was under the KL, at ~100 epochs, and against a different
#:   attention path; carrying it over would fight the architecture parity this round is for.
#:   Freezing L is also what makes MASK DEPTH a legal factor -- layer 3 does not exist at
#:   L=2, so the two axes cannot be crossed.
FIXED: dict[str, dict[str, Any]] = {
    "betaxanthin": {
        "node_embeddings": "prot_T5_all",
        "lr": 4.06e-4,  # _003 screen winner (0.4050)
        "weight_decay": 2.72e-6,  # same trial
        "dist": "crps",  # _002 best on this arm (0.4301)
        "target_norm": "zscore",
        "dropout": 0.1,
        "perturbation_num_heads": 6,
        # Frozen default; overridden by the factor on the arms that sweep it.
        "mask_layers": [1],
    },
    "beta_carotene": {
        "node_embeddings": "prot_T5_all",
        "lr": 1.81e-4,
        "weight_decay": 2.11e-7,
        "dist": "quantile",
        "target_norm": "zscore",
        "dropout": 0.1,
        "perturbation_num_heads": 6,
        # Frozen default; overridden by the factor on the arms that sweep it.
        "mask_layers": [1],
    },
    "mulleder19": {
        "node_embeddings": "prot_T5_all",
        "lr": 4.56e-4,
        "weight_decay": 2.11e-5,
        "dist": "quantile",  # _002 best; `energy` is this arm's own factor, below
        "target_norm": "zscore",
        "dropout": 0.1,
        "perturbation_num_heads": 6,
        # Frozen default; overridden by the factor on the arms that sweep it.
        "mask_layers": [1],
        # Rank of V in Sigma = diag(sigma^2) + V V^T. 8 fits inside F = 19 (32 cannot be
        # realized by a 19x19 covariance and `DistHead` has no guard). Inert unless
        # dist == energy.
        "energy_rank": 8,
    },
}
FIXED["bx_ctrl"] = dict(FIXED["betaxanthin"])
FIXED["bx_m19"] = dict(FIXED["betaxanthin"])

#: TWO FACTORS x TWO INIT SEEDS = the 8 runs. The split is now PINNED (`split_seed: 0`), so
#: `seed` varies weight initialization ONLY -- which is what makes replication worth paying
#: for at this sample size. 019 measured between-seed sd 0.0444 against a within-seed
#: across-arm sd of 0.0058 when one knob drove both; separating them is what turns two runs
#: into a paired comparison rather than two draws from a wide nuisance distribution.
#:
#: `hadamard` {off, replace} -- THE PAIR TERM, and on a deletion panel it is the sharpest
#:   architectural question we have. 019 proved the additive operator has NO pair term at
#:   |S_b| = 1: the cross-attention softmax runs over a one-element key set, so alpha == 1
#:   and H_pert[b,i] = g(h_i + c_b) with c_b shared by every gene i. Re-drawing W_Q, W_K at
#:   std 10 changed the output by exactly 0.0 -- 16,200 of 32,760 attention parameters are
#:   dead. EVERY metabolism strain is a single deletion, so this holds for all four arms
#:   without exception. `replace` swaps in h_i * (1 + gamma(c_b)), a genuine rank-d = 90
#:   bilinear interaction, identity at init.
#: `mask_layers` {[1], [1,3]} -- MASK DEPTH, wave 5's own open stage. Layer 1 is where the
#:   graph enters today; adding layer 3 routes it again deeper in the encoder. Legal only
#:   because L is frozen at 6 -- at L=2 layer 3 does not exist and the config RAISES.
FACTORS: dict[str, dict[str, list[Any]]] = {
    "betaxanthin": {"hadamard": ["off", "replace"], "mask_layers": [[1], [1, 3]]},
    "beta_carotene": {"hadamard": ["off", "replace"], "mask_layers": [[1], [1, 3]]},
    #: mulleder19 swaps MASK DEPTH for the DISTRIBUTION -- the only arm where that is a real
    #: question, because it is the only one with a joint distribution to model. At F = 1 the
    #: predictive covariance Sigma = diag(sigma^2) + V V^T has V of shape [1, k], so V V^T is
    #: a scalar and `energy` degenerates into a noisier Monte-Carlo CRPS. Here F = 19. 019's
    #: "the distributional axis is not the lever" was measured on expression, read by a
    #: per-gene INDEPENDENT head -- a statement about MARGINALS, which says nothing about
    #: whether modelling the joint pays.
    "mulleder19": {"hadamard": ["off", "replace"], "dist": ["quantile", "energy"]},
}
#: The controlled pair runs TWO settings per side, so the 8 runs are
#: 2 settings x {control, joint} x 2 init seeds. The pairing IS the experiment and it is
#: worth the factor budget: both arms share every (setting, seed) cell, hence the same
#: pinned split AND the same initialization, so `bx_m19 - bx_ctrl` is a PAIRED difference.
#: Mask depth is frozen at [1] -- an axis that interacted with the auxiliary head would
#: confound the one difference under test.
FACTORS["bx_ctrl"] = {"hadamard": ["off", "replace"]}
FACTORS["bx_m19"] = dict(FACTORS["bx_ctrl"])

#: Depth, frozen everywhere at the `_012` value. Left explicit rather than defaulted so a
#: frozen architecture is always a stated choice.
FIXED_LAYERS = {
    a: 6 for a in ("betaxanthin", "beta_carotene", "mulleder19", "bx_ctrl", "bx_m19")
}

#: Short factor names for the setting label (a W&B tag, and a run-list column).
ABBREV = {"dropout": "d", "num_transformer_layers": "L"}


def settings() -> list[dict[str, Any]]:
    """Enumerate the crossed factor levels, in a fixed order, as named settings.

    The name is self-describing but carries NO `=`: it travels into `wandb.tags` through a
    Hydra override, and `=` inside a list element is a parse error in Hydra's override
    grammar ("no viable alternative at input"), which fails at compose time rather than at
    run time -- i.e. it would kill every trial of the job.
    """
    factors = FACTORS[ARM]
    keys = list(factors)
    out = []
    for i, levels in enumerate(itertools.product(*(factors[k] for k in keys))):
        params = dict(zip(keys, levels, strict=True))
        # Built from whatever factors THIS arm has, rather than a fixed template: mulleder19
        # swaps depth for `dist`, so a hardcoded name would either KeyError or silently label
        # two different settings identically. Abbreviated because the name becomes a W&B tag
        # and a run-list column, where a 60-character label is unreadable.
        parts = "_".join(
            f"mask{''.join(str(x) for x in v)}"
            if isinstance(v, list)
            else (f"{ABBREV.get(k, k)}{v:g}" if isinstance(v, (int, float)) else f"{v}")
            for k, v in sorted(params.items())
        )
        params["name"] = f"s{i}_{parts}"
        out.append(params)
    return out


SETTINGS = settings()
SETTING_NAMES = [s["name"] for s in SETTINGS]
SETTING_BY_NAME = {s["name"]: s for s in SETTINGS}


def _overrides(
    setting: dict[str, Any], seed: int, max_time_s: float | None
) -> list[str]:
    """Translate one (setting, seed) cell into Hydra overrides.

    A factor's level takes precedence over the arm's frozen value for the same key, so an axis
    can move between FACTORS and FIXED (as `dist` and depth do across arms) without this
    function needing to know which arm it is serving.
    """
    fixed = dict(FIXED[ARM])
    fixed["num_transformer_layers"] = FIXED_LAYERS.get(ARM, 0)
    fixed.update({k: v for k, v in setting.items() if k != "name"})
    heads = ",".join(ACTIVE_HEADS)
    # Normalization is applied to EVERY active head, so the joint arm standardizes the
    # metabolome too -- an un-standardized 19-AA target would let a few large-magnitude amino
    # acids dominate the auxiliary loss and change what the shared encoder is pulled toward.
    zscore = fixed["target_norm"] == "zscore"
    normalize_list = [] if zscore else list(ACTIVE_HEADS)
    standardize_list = list(ACTIVE_HEADS) if zscore else []
    ov = [
        f"seed={seed}",
        f"multitask.active_heads=[{heads}]",
        f"cell_dataset.node_embeddings=[{fixed['node_embeddings']}]",
        "model.learnable_embedding.enabled=false",
        "multitask.decoder=s1_pool",
        f"multitask.dist={fixed['dist']}",
        f"multitask.normalize_vector_targets=[{','.join(normalize_list)}]",
        f"multitask.standardize_per_feature_target=[{','.join(standardize_list)}]",
        f"model.num_transformer_layers={fixed['num_transformer_layers']}",
        f"model.dropout={fixed['dropout']}",
        f"model.perturbation_head.num_heads={fixed['perturbation_num_heads']}",
        # THE PAIR TERM. `off` is the additive reference operator, which at |S_b| = 1 has no
        # pair-(p,i) term at all; `replace` swaps in h_i * (1 + gamma(c_b)), a rank-d = 90
        # bilinear interaction that is the exact identity at init.
        f"model.perturbation_head.hadamard={fixed['hadamard']}",
        # Encoder layers where the nine-relation hard mask is applied. NOT a lambda: the mask
        # is a structural constraint, so there is nothing to calibrate -- only where to apply
        # it. Layer indices must exist at `num_transformer_layers`, which is why L is frozen.
        f"model.attention_mask.layers=[{','.join(str(x) for x in fixed['mask_layers'])}]",
        f"regression_task.optimizer.lr={fixed['lr']}",
        f"regression_task.optimizer.weight_decay={fixed['weight_decay']}",
        f"data_module.num_workers={os.getenv('NUM_WORKERS', '4')}",
        # QUOTED elements. A tag like `s0_0.1_lam1.6e-05_2` contains `.` and `-`, which
        # Hydra's override grammar tries to lex as a number inside a list; quoting makes each
        # element an unambiguous string.
        f"wandb.tags=['delta','grid','{ARM}','{setting['name']}','seed-{seed}']",
    ]
    if "energy_rank" in fixed:
        ov.append(f"multitask.energy_rank={fixed['energy_rank']}")
    if max_time_s is not None:
        ov.append(f"trainer.max_time_s={max_time_s:.0f}")
    return ov


def _trial_time_budget() -> float | None:
    """Seconds of TRAINING this trial gets, from the job's remaining wall clock.

    Computed at CLAIM time rather than once per job: the eight round-0 runs all start
    together and so all get (almost) the full window, while a replicate that starts later
    correctly gets only what is left. `TEARDOWN_S` is subtracted so the test pass, the
    prediction dump and the wandb teardown happen INSIDE the allocation -- the graceful stop
    is worth nothing if slurm kills the process during the work the stop exists to protect.
    """
    if DEADLINE is None:
        return None
    return max(0.0, DEADLINE - time.time() - TEARDOWN_S)


def _record(trial: optuna.Trial, metrics: dict[str, float]) -> None:
    """Mirror the diagnostics into the trial so the study reads standalone.

    Recorded, never ranked. `pred_sd_ratio` is the mean-collapse diagnostic for a SCALAR head
    (`pearson_per_instance` is undefined at F = 1); `peak_epoch` / `n_val_epochs` are the
    covariates for the max-over-epochs bias, which on `_002` mulleder19 correlated with the
    objective at r = +0.75; the `test/*` keys exist only on arms running the pinned split.
    """
    for name, key in (
        ("pearson", f"val/{RANK_PHENO}/pearson_per_feature_max"),
        ("spearman", f"val/{RANK_PHENO}/spearman_per_feature_max"),
        ("ranked_smooth3", RANKED_SMOOTH3),
        ("pred_sd_ratio", f"val/{RANK_PHENO}/pred_sd_ratio_at_peak"),
        ("graph_reg_ratio", "val/graph_reg/ratio_to_data_at_peak"),
        ("n_val_epochs", "val/n_val_epochs"),
        ("peak_epoch", "val/peak_epoch"),
        ("test_pearson", f"test/{RANK_PHENO}/pearson_per_feature"),
        ("test_spearman", f"test/{RANK_PHENO}/spearman_per_feature"),
    ):
        trial.set_user_attr(name, metrics.get(key))
    # The auxiliary head's OWN score. "The metabolome helped betaxanthin" and "the metabolome
    # head learned anything" are different claims, and the second explains the first -- an
    # auxiliary head at r ~ 0 that still moves the primary metric would mean the gain came
    # from regularization, not from shared metabolic signal.
    for aux in ACTIVE_HEADS:
        if aux == RANK_PHENO:
            continue
        trial.set_user_attr(
            f"aux_{aux}_pearson", metrics.get(f"val/{aux}/pearson_per_feature_max")
        )
    for key in ("coverage_50", "coverage_80", "pit_ks"):
        trial.set_user_attr(
            f"calib_{key}", metrics.get(f"val/{RANK_PHENO}/calib/{key}_at_peak")
        )


def objective(trial: optuna.Trial) -> float:
    """Run one (setting, seed) cell.

    Both parameters are `suggest_categorical` over the FULL level set even though every trial
    is enqueued: optuna replays an enqueued trial by re-suggesting, and a suggest whose
    choices differ from the enqueued value's domain is a distribution mismatch, not a
    fixed value.
    """
    name = trial.suggest_categorical("setting", SETTING_NAMES)
    seed = trial.suggest_categorical("seed", SEEDS)
    setting = SETTING_BY_NAME[name]
    budget = _trial_time_budget()
    with initialize_config_dir(version_base=None, config_dir=CONF_DIR):
        cfg = compose(
            config_name=BASE_CONFIG, overrides=_overrides(setting, seed, budget)
        )
    print(
        f"[{ARM} w{WORKER_ID}] trial {trial.number}: {name} seed={seed} "
        f"emb={FIXED[ARM]['node_embeddings']} lr={FIXED[ARM]['lr']:.3g} "
        f"wd={FIXED[ARM]['weight_decay']:.3g} "
        f"budget={'unbounded' if budget is None else f'{budget / 3600:.2f} h'}",
        flush=True,
    )
    print(OmegaConf.to_yaml(cfg.multitask), flush=True)
    metrics: dict[str, float] = run_training(cfg)
    torch.cuda.empty_cache()
    _record(trial, metrics)
    trial.set_user_attr("setting_params", {k: v for k, v in setting.items()})
    trial.set_user_attr("train_time_budget_s", budget)
    if OBJECTIVE_METRIC not in metrics:
        raise optuna.TrialPruned(
            f"{OBJECTIVE_METRIC} not logged (keys: {sorted(metrics)[:12]}...)"
        )
    return float(metrics[OBJECTIVE_METRIC])


def get_study() -> optuna.Study:
    """Create-or-load. The study is a WORK LEDGER, not a search.

    Every trial is enqueued, so the sampler is never consulted for anything that matters; it
    exists because `study.optimize` requires one. Optuna is used here for what it is genuinely
    good at in this setting -- concurrency-safe claiming across four processes on one SQLite
    file, durable per-trial attributes, and resume-after-kill -- and for nothing else.
    """
    return optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE,
        sampler=optuna.samplers.RandomSampler(seed=0),
        load_if_exists=True,
        direction="maximize",
    )


def _enqueue_all(study: optuna.Study) -> int:
    """Enqueue every (setting, seed) cell, ROUND-MAJOR, skipping what already exists.

    Round-major is load-bearing: a job killed by the wall clock then leaves every setting with
    the SAME seed count, so the factorial stays balanced and the comparison stays valid.
    Seed-major would leave 12 seeds on two settings and none on six.

    Idempotent -- a requeue after a kill tops the queue back up rather than duplicating it.
    An enqueued (WAITING) trial keeps its values in `system_attrs["fixed_params"]` until it
    runs, so `.params` alone would fail to recognize the existing queue.
    """
    seen = set()
    for t in study.trials:
        params = t.params or t.system_attrs.get("fixed_params", {})
        if "setting" in params and "seed" in params:
            seen.add((params["setting"], params["seed"]))
    added = 0
    for r in range(ROUNDS):
        seed = SEEDS[r % len(SEEDS)]
        for name in SETTING_NAMES:
            if (name, seed) in seen:
                continue
            study.enqueue_trial({"setting": name, "seed": seed})
            seen.add((name, seed))
            added += 1
    return added


def _write_manifest() -> None:
    """Record the grid as an artifact, so a result can be read without re-reading this file."""
    out_dir = os.getenv("GRID_MANIFEST_DIR")
    if not out_dir:
        return
    os.makedirs(out_dir, exist_ok=True)
    path = osp.join(out_dir, f"grid_manifest_{ARM}.json")
    with open(path, "w") as fh:
        json.dump(
            {
                "arm": ARM,
                "base_config": BASE_CONFIG,
                "study_name": STUDY_NAME,
                "objective_metric": OBJECTIVE_METRIC,
                "active_heads": ACTIVE_HEADS,
                "fixed": FIXED[ARM],
                "factors": FACTORS[ARM],
                "settings": SETTINGS,
                "rounds": ROUNDS,
                "seeds": SEEDS[:ROUNDS],
            },
            fh,
            indent=2,
        )
    print(f"[grid] wrote {path}", flush=True)


def main() -> None:
    study = get_study()
    if "--create-only" in sys.argv:
        added = _enqueue_all(study)
        print(
            f"[create-only] study={STUDY_NAME} arm={ARM} "
            f"{len(SETTINGS)} settings x {ROUNDS} rounds -> {added} newly enqueued "
            f"({len(study.trials)} total in study)",
            flush=True,
        )
        for s in SETTINGS:
            print(f"  {s['name']}", flush=True)
        _write_manifest()
        return

    # Stop CLAIMING new work once too little wall clock remains to say anything. A trial that
    # DOES start is bounded by `trainer.max_time_s` rather than by slurm, so it always ends
    # gracefully -- this guard is about not starting a run too short to read, not about
    # avoiding a kill.
    def deadline_callback(st: optuna.Study, _: optuna.trial.FrozenTrial) -> None:
        if DEADLINE is not None and time.time() > DEADLINE - MIN_TRIAL_S - TEARDOWN_S:
            print(
                f"[{ARM} w{WORKER_ID}] less than "
                f"{(MIN_TRIAL_S + TEARDOWN_S) / 60:.0f} min of usable wall clock left "
                f"-- not claiming further work.",
                flush=True,
            )
            st.stop()

    if DEADLINE is not None and time.time() > DEADLINE - MIN_TRIAL_S - TEARDOWN_S:
        print(f"[{ARM} w{WORKER_ID}] past the deadline before starting -- exiting.")
        return

    waiting = sum(1 for t in study.trials if t.state.name == "WAITING")
    print(
        f"[{ARM} w{WORKER_ID}] study={STUDY_NAME} objective={OBJECTIVE_METRIC} "
        f"{waiting} trials waiting, base={BASE_CONFIG}",
        flush=True,
    )
    # `n_trials` bounds this worker; the real stop is the empty queue or the deadline. optuna
    # runs an enqueued trial ahead of a sampled one, so the queue drains before any random
    # config could be invented -- and `_enqueue_all` sizes the queue to the grid exactly.
    study.optimize(
        objective,
        n_trials=len(SETTINGS) * ROUNDS,
        catch=(Exception,),
        callbacks=[deadline_callback],
    )

    complete = [t for t in study.trials if t.state.name == "COMPLETE" and t.values]
    if WORKER_ID == 0 and complete:
        by_setting: dict[str, list[float]] = {}
        for t in complete:
            by_setting.setdefault(t.params["setting"], []).append(t.values[0])
        print(f"\n[{ARM}] {len(complete)} completed runs")
        for name in SETTING_NAMES:
            vals = by_setting.get(name, [])
            if not vals:
                print(f"  {name:56s} n=0")
                continue
            mean = sum(vals) / len(vals)
            sd = (
                (sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5
                if len(vals) > 1
                else float("nan")
            )
            sem = sd / len(vals) ** 0.5 if len(vals) > 1 else float("nan")
            print(
                f"  {name:56s} n={len(vals):2d} mean={mean:.4f} sd={sd:.4f} sem={sem:.4f}"
            )


if __name__ == "__main__":
    import multiprocessing as mp

    mp.set_start_method("spawn", force=True)
    main()
