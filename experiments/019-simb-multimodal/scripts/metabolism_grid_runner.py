# experiments/019-simb-multimodal/scripts/metabolism_grid_runner.py
# [[experiments.019-simb-multimodal.scripts.metabolism_grid_runner]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/metabolism_grid_runner
"""EIGHT FIXED SETTINGS x AS MANY SEEDS AS THE WALL CLOCK BUYS -- the Delta metabolism runs.

This REPLACES the Optuna search (`optuna_metabolism_sweep.py`) for the Delta jobs, and the
reason is a measurement, not a preference.

WHY NOT ANOTHER SEARCH
----------------------
The scalar metabolism arms have a REPLICATE NOISE of sigma = 0.030 (betaxanthin) and 0.025
(beta-carotene) on `pearson_per_feature` -- measured directly, from `_002` trials that
re-proposed identical hyperparameters AND an identical seed:

    betaxanthin    trials  3/10/11   0.4216 / 0.4095 / 0.3584     sigma = 0.030
    beta_carotene  trials 18/22      0.1993 / 0.1494              sigma = 0.025

`_002`'s top FIVE betaxanthin configs spanned 0.021 -- 0.68 sigma. The search has already
told us everything a single-seed search can tell us: it can rank AXES, and it cannot pick a
WINNER. Running a third search would buy more unreadable trials. What is missing is
REPLICATION, and replication is exactly what a 2-day 4-GPU node is for.

    8 settings x R seeds, R growing until the deadline.
    Per setting, SE = sigma / sqrt(R):  R=5 -> 0.013,  R=12 -> 0.0087.
    Per FACTOR MAIN EFFECT (4 settings vs 4, so 4R runs a side):
    SE = sigma * sqrt(2 / (4R)):        R=12 -> 0.0035.

That resolves differences an order of magnitude smaller than any number the searches could
separate, and it is what makes "we beat X" a statement with an error bar on it.

THE 8 SETTINGS ARE A 2x2x2 FACTORIAL, not eight assorted guesses
----------------------------------------------------------------
Every arm fixes the axes its own measurements already settled and varies three binary
factors, fully crossed. Fully crossed matters: each factor's main effect is then estimated
from ALL 8R runs rather than from one pair, and the interactions are visible instead of
confounded. That is the whole design -- eight "similar settings", differing in three bits.

RUNTIME MODEL
-------------
The work list is enqueued ROUND-MAJOR -- round r contributes one seed to EACH of the 8
settings. A job killed at the wall clock therefore leaves every setting with the SAME number
of seeds, and the comparison stays balanced. Enqueueing seed-major (all seeds of setting 0,
then setting 1, ...) would leave a truncated run with 12 seeds on two settings and none on
six, i.e. no comparison at all.

Workers stop CLAIMING new work once `GRID_DEADLINE_EPOCH - GRID_TRIAL_MARGIN_S` has passed,
so a run in flight is not killed mid-training by the slurm wall clock. Everything already
completed is durable in the study, and a requeue resumes from it.

Environment (set by the slurm launcher):
    ARM                    betaxanthin | mulleder19 | beta_carotene | bx_ctrl | bx_m19
    GRID_CONF_DIR          absolute path to the arm's Hydra conf dir      (required)
    GRID_BASE_CONFIG       Hydra base config name                        (required)
    OPTUNA_STORAGE         sqlite:///.../optuna_<exp>_<arm>_grid.db       (required)
    OPTUNA_STUDY_NAME      default: <arm>_grid_000
    GRID_ROUNDS            seed rounds to enqueue (default 12)
    GRID_DEADLINE_EPOCH    unix seconds; stop claiming after this (default: no deadline)
    GRID_TRIAL_MARGIN_S    reserve for one trial (default 5400 = 90 min)
    OPTUNA_WORKER_ID       0..N-1, for logging only
    NUM_WORKERS            dataloader workers per trial
    GRID_MANIFEST_DIR      where --create-only writes grid_manifest_<arm>.json (optional)

Modes:
    --create-only   create the study, enqueue every (setting, seed), write the manifest.
                    Run ONCE from the launcher before the workers, so a fresh SQLite file's
                    DDL is not raced by four processes that each spend >15s importing torch.
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
ROUNDS = int(os.getenv("GRID_ROUNDS", "12"))
WORKER_ID = int(os.getenv("OPTUNA_WORKER_ID", "0"))
DEADLINE = float(os.getenv("GRID_DEADLINE_EPOCH", "0")) or None
TRIAL_MARGIN_S = float(os.getenv("GRID_TRIAL_MARGIN_S", "5400"))

#: Seeds, in the order rounds consume them. `cfg.seed` drives BOTH model init AND the
#: CellDataModule split, so a round is a fresh SPLIT as well as a fresh init -- i.e. this is
#: repeated-random-subsampling validation, which estimates "best in expectation over splits",
#: the quantity we actually want to select on. 42 leads so round 0 is directly comparable to
#: the whole `_002`/`_003`/`_004` history, all of which ran at 42.
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
FIXED: dict[str, dict[str, Any]] = {
    "betaxanthin": {
        "node_embeddings": "prot_T5_all",
        "num_transformer_layers": 2,  # _002: 2 -> 0.288, 4 -> 0.197, 6 -> 0.124
        "lr": 4.06e-4,  # _003 screen winner (0.4050)
        "weight_decay": 2.72e-6,  # same trial
        "dropout": 0.1,
        "perturbation_num_heads": 6,
    },
    "beta_carotene": {
        "node_embeddings": "prot_T5_all",
        "num_transformer_layers": 4,  # _002: this arm preferred 4, unlike betaxanthin
        "lr": 1.81e-4,
        "weight_decay": 2.11e-7,
        "dropout": 0.1,
        "perturbation_num_heads": 6,
    },
    "mulleder19": {
        "node_embeddings": "prot_T5_all",
        "num_transformer_layers": 4,  # _002 best (0.1798) and _004 best (0.1713) agree
        "lr": 4.56e-4,
        "weight_decay": 2.11e-5,
        "dropout": 0.1,
        "perturbation_num_heads": 6,
        # Rank of V in Sigma = diag(sigma^2) + V V^T. 8 fits inside F = 19 (32 could not be
        # realized and `DistHead` has no guard). Inert unless dist == energy.
        "energy_rank": 8,
    },
}
FIXED["bx_ctrl"] = dict(FIXED["betaxanthin"])
FIXED["bx_m19"] = dict(FIXED["betaxanthin"])

#: THE THREE FACTORS, fully crossed -> the 8 settings. Chosen so BOTH levels of every factor
#: have real support in our own measurements; a factor whose second level nobody believes is
#: a wasted half of the grid.
#:
#: `dist`   -- crps won `_002` on betaxanthin (0.4301) while nll won `_003` (0.4050); on
#:             mulleder19 quantile won `_002` (0.1798) and energy won `_004` (0.1713). Point
#:             (MSE) stays excluded: its optimum IS the conditional mean, i.e. the
#:             mean-collapse the honest metric punishes, and it placed last on all three arms.
#: `target_norm` -- `_002`'s betaxanthin winner was yeo_johnson, `_003`'s was zscore. Straight
#:             contradiction between rounds, never replicated. Exactly what a factorial is for.
#: `graph_reg_lambda` -- MEASURED per arm from the `_002` `graph_reg_ratio` attrs (graph term /
#:             data term): parity (ratio = 1) sits at 1.6e-4 / 3.4e-4 / 2.2e-4 on betaxanthin /
#:             beta-carotene / mulleder19. A shared lambda is NOT a shared prior strength, so
#:             each arm brackets its OWN parity by a decade. The lambda = 0 ablation is
#:             deliberately NOT a level here: `_002` already answered it for betaxanthin
#:             (0.113 off vs 0.280 on), and spending half this grid re-answering it would cost
#:             the resolution that is the entire point of running it.
FACTORS: dict[str, dict[str, list[Any]]] = {
    "betaxanthin": {
        "dist": ["crps", "nll"],
        "target_norm": ["zscore", "yeo_johnson"],
        "graph_reg_lambda": [1.6e-5, 1.6e-4],
    },
    "beta_carotene": {
        "dist": ["crps", "quantile"],
        "target_norm": ["zscore", "yeo_johnson"],
        "graph_reg_lambda": [3.4e-5, 3.4e-4],
    },
    "mulleder19": {
        "dist": ["quantile", "energy"],
        "target_norm": ["zscore", "yeo_johnson"],
        "graph_reg_lambda": [2.2e-5, 2.2e-4],
    },
}
#: The controlled pair runs FOUR settings per side, so the 8 runs of round r are
#: 4 settings x {control, joint} -- the pairing is the experiment, so it buys the third
#: factor's worth of budget. `target_norm` is frozen at zscore: it is the normalization the
#: metabolome head also needs, and a norm that differed across the pair would be a second
#: difference between arms that are supposed to differ in exactly one thing.
FACTORS["bx_ctrl"] = {
    "dist": ["crps", "nll"],
    "target_norm": ["zscore"],
    "graph_reg_lambda": [1.6e-5, 1.6e-4],
}
FACTORS["bx_m19"] = dict(FACTORS["bx_ctrl"])


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
        params["name"] = (
            f"s{i}_{params['dist']}_{params['target_norm']}"
            f"_lam{params['graph_reg_lambda']:g}"
        )
        out.append(params)
    return out


SETTINGS = settings()
SETTING_NAMES = [s["name"] for s in SETTINGS]
SETTING_BY_NAME = {s["name"]: s for s in SETTINGS}


def _overrides(setting: dict[str, Any], seed: int) -> list[str]:
    """Translate one (setting, seed) cell into Hydra overrides."""
    fixed = FIXED[ARM]
    heads = ",".join(ACTIVE_HEADS)
    # Normalization is applied to EVERY active head, so the joint arm standardizes the
    # metabolome too -- an un-standardized 19-AA target would let a few large-magnitude amino
    # acids dominate the auxiliary loss and change what the shared encoder is pulled toward.
    zscore = setting["target_norm"] == "zscore"
    normalize_list = [] if zscore else list(ACTIVE_HEADS)
    standardize_list = list(ACTIVE_HEADS) if zscore else []
    ov = [
        f"seed={seed}",
        f"multitask.active_heads=[{heads}]",
        f"cell_dataset.node_embeddings=[{fixed['node_embeddings']}]",
        "model.learnable_embedding.enabled=false",
        "multitask.decoder=s1_pool",
        f"multitask.dist={setting['dist']}",
        f"multitask.normalize_vector_targets=[{','.join(normalize_list)}]",
        f"multitask.standardize_per_feature_target=[{','.join(standardize_list)}]",
        f"model.num_transformer_layers={fixed['num_transformer_layers']}",
        f"model.dropout={fixed['dropout']}",
        f"model.perturbation_head.num_heads={fixed['perturbation_num_heads']}",
        # The nine per-head `lambda` entries interpolate this single scalar, so overriding it
        # retunes all nine regularized heads at once.
        f"model.graph_regularization.graph_reg_lambda={setting['graph_reg_lambda']}",
        f"regression_task.optimizer.lr={fixed['lr']}",
        f"regression_task.optimizer.weight_decay={fixed['weight_decay']}",
        f"data_module.num_workers={os.getenv('NUM_WORKERS', '4')}",
        # QUOTED elements. A tag like `s0_crps_zscore_lam1.6e-05` contains `.` and `-`, which
        # Hydra's override grammar tries to lex as a number inside a list; quoting makes each
        # element an unambiguous string.
        f"wandb.tags=['delta','grid','{ARM}','{setting['name']}','seed-{seed}']",
    ]
    if "energy_rank" in fixed:
        ov.append(f"multitask.energy_rank={fixed['energy_rank']}")
    return ov


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
    with initialize_config_dir(version_base=None, config_dir=CONF_DIR):
        cfg = compose(config_name=BASE_CONFIG, overrides=_overrides(setting, seed))
    print(
        f"[{ARM} w{WORKER_ID}] trial {trial.number}: {name} seed={seed} "
        f"emb={FIXED[ARM]['node_embeddings']} L={FIXED[ARM]['num_transformer_layers']} "
        f"lr={FIXED[ARM]['lr']:.3g} wd={FIXED[ARM]['weight_decay']:.3g}",
        flush=True,
    )
    print(OmegaConf.to_yaml(cfg.multitask), flush=True)
    metrics: dict[str, float] = run_training(cfg)
    torch.cuda.empty_cache()
    _record(trial, metrics)
    trial.set_user_attr("setting_params", {k: v for k, v in setting.items()})
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

    # Stop CLAIMING new work once there is not a full trial's worth of wall clock left, so the
    # slurm kill never lands mid-training and throws away a run that was nearly finished.
    def deadline_callback(st: optuna.Study, _: optuna.trial.FrozenTrial) -> None:
        if DEADLINE is not None and time.time() > DEADLINE - TRIAL_MARGIN_S:
            print(
                f"[{ARM} w{WORKER_ID}] within {TRIAL_MARGIN_S / 60:.0f} min of the "
                f"deadline -- stopping after this trial.",
                flush=True,
            )
            st.stop()

    if DEADLINE is not None and time.time() > DEADLINE - TRIAL_MARGIN_S:
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
