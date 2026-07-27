# experiments/020-cachera-betaxanthin/scripts/optuna_metabolism_sweep.py
# [[experiments.020-cachera-betaxanthin.scripts.optuna_metabolism_sweep]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/020-cachera-betaxanthin/scripts/optuna_metabolism_sweep
"""Optuna sweep driver for the three single-target metabolism arms.

ONE ARM PER GPU, three concurrent studies. Each arm is a SEPARATE study with its OWN
objective metric, because the three targets are not commensurable:

    ARM=betaxanthin    val/betaxanthin/pearson_per_feature_max     ceiling r    = 0.914
    ARM=beta_carotene  val/beta_carotene/spearman_per_feature_max  ceiling rho ~ 0.54
    ARM=mulleder19     val/mulleder19/pearson_per_feature_max      ceiling unmeasured

Beta-carotene is ranked on SPEARMAN and that is not a preference: Ozaydin's readout is a
SUBJECTIVE ORDINAL on -5..+5, so a Pearson on it asserts an interval scale the measurement
does not have. Ranking the sweep on Pearson would select models that fit a scale artifact.

PEAK, not last epoch (`_max` suffix, supplied by BestMetricTracker). These runs reach a
per-feature Pearson peak and then MSE-collapse toward the per-feature mean, so the last
epoch systematically understates. Read every peak ALONGSIDE the val loss at that epoch: a
peak with a flat loss is a maximum of noise, a peak with a loss drop is a fit. Job 1341 is
the worked example -- peak 0.323 at val loss 0.885, the only point below 0.971 (the variance
of the z-scored target).

WHAT THIS INHERITS FROM THE EXPRESSION / MORPHOLOGY ROUNDS, AND WHAT HAD TO CHANGE
----------------------------------------------------------------------------------
Inherited verbatim from `optuna_joint_sweep.py` (_005): the coverage-audited node-embedding
axis, the distributional axis, the hp profiles, per-head target norms, multi-worker
create-or-load with a serialized `--create-only` pre-create, and ranking on the
phenotype-namespaced `_max` metric.

Three things could NOT be carried over as-is:

* **hidden_channels is {90, 180}, not {128, 180}.** With the full nine-graph block there is
  one regularized attention head per graph, so `num_attention_heads = 9` and hidden MUST be
  divisible by 9. 128 is not (128/9 = 14.2). This is the hidden-dim fix the expression sweep
  needed once it moved past two graphs. Both 90 and 180 also divide by the perturbation
  head's 6.

* **graph_reg_lambda is RE-SCALED by ~15x downward.** The old grid {0, 3e-4, 1e-3, 3e-3} was
  tuned while lambda was being applied TWICE -- per-graph and then globally from the same
  config key -- so the effective weight was lambda^2, i.e. 1e-6 at a nominal 1e-3. With that
  fixed, a nominal 1e-3 puts the graph prior at 3.85x the DATA loss and training is governed
  by the prior. The grid below brackets 10 / 25 / 50 % of the data loss, measured on a real
  batch (6.5e-5 -> val/graph_reg/ratio_to_data = 0.251). Carrying the old numbers over would
  have silently swamped every trial.

The DNA embedding axes (`fudt_upstream`, `nt_window_5979`) are dropped on evidence, not
taste: in the kNN embedding probe every nucleotide representation landed on the `random_100`
control at 2560-5120 dim, while protein representations carried real signal. Trials are
scarce at ~1-2 h each, so the axis is spent on options that measured above the control.

Environment (set by the slurm launcher):
    ARM                 betaxanthin | beta_carotene | mulleder19   (required)
    OPTUNA_STORAGE      sqlite:////<scratch>/.../optuna_020_<arm>.db   (required)
    OPTUNA_STUDY_NAME   default: <arm>_000
    OPTUNA_N_TRIALS     trials THIS worker runs (default: 40)
    OPTUNA_WORKER_ID    seeds the sampler (default: 0)
    METAB_BASE_CONFIG   base Hydra config (default: <arm>)
    NUM_WORKERS         dataloader workers per trial (default: 4)
"""

import os
import os.path as osp
import sys

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
BASE_CONFIG = os.getenv("METAB_BASE_CONFIG", ARM)
STORAGE = os.environ["OPTUNA_STORAGE"]
STUDY_NAME = os.getenv("OPTUNA_STUDY_NAME", f"{ARM}_000")
N_TRIALS = int(os.getenv("OPTUNA_N_TRIALS", "40"))
WORKER_ID = int(os.getenv("OPTUNA_WORKER_ID", "0"))

#: Ranking metric per arm. Beta-carotene is an ordinal -> Spearman. See module docstring.
OBJECTIVE_METRIC = {
    "betaxanthin": "val/betaxanthin/pearson_per_feature_max",
    "beta_carotene": "val/beta_carotene/spearman_per_feature_max",
    "mulleder19": "val/mulleder19/pearson_per_feature_max",
}[ARM]

#: NO DECODER AXIS -- verified, not assumed. `multitask.decoder` switches the STRUCTURAL form
#: of the inherited `global` (morphology) head ONLY (`train_cgt_multitask.py:34,160`). All
#: three metabolism targets are read by the metabolism subclass's OWN heads --
#: `ProductScalarHead` and `MetabolomeVectorHead` -- which are plain MLPs and never consult
#: it (grep for `decoder` in `cell_graph_transformer_metabolism.py` returns only prose).
#:
#: Sweeping it anyway would be a PHANTOM DIMENSION: it splits the TPE search space in two
#: without changing the model, so half the trial budget would buy duplicate configurations.
#: That is the exact trap the expression sweep documents for its own arm.
#:
#: FOLLOW-UP, deliberately not done here: a true S3 cross-attention decoder for the 19-AA
#: metabolome (one learned query per amino acid, each reading its own mixture of tokens) does
#: not exist yet. If the mulleder19 arm stays flat, building that is the next structural
#: lever -- it is the S1-pool dilution argument applied to a 19-feature vector.
DECODER = "s1_pool"

PROFILES = {
    "baseline": {"lr": 3.0e-4, "dropout": 0.1, "weight_decay": 1.0e-8},
    "aggressive": {"lr": 1.0e-3, "dropout": 0.0, "weight_decay": 1.0e-4},
    # Added for this round: the 019 lesson is that on a few thousand genotypes this
    # architecture needs to be small AND decayed. wd 1e-6 memorized (train 0.86 / val 0.02).
    "decayed": {"lr": 1.0e-3, "dropout": 0.1, "weight_decay": 1.0e-4},
}

#: Coverage-audited: all 6607 genes, 0 missing, 0 zero-vectors. The `no_dubious` /
#: `no_uncharacterized` variants are deliberately absent -- they emit 684/668 ALL-ZERO
#: vectors, i.e. silently information-free for exactly the hardest ORFs. Nucleotide axes are
#: dropped on measured evidence (kNN probe: indistinguishable from random_100).
NODE_EMBEDDINGS = [
    "prot_T5_all",
    "esm2_t33_650M_UR50D_all",
    "calm",
    "codon_frequency",
    "random_100",  # CONTROL: unique but meaningless -> isolates CONTENT from IDENTITY
]

#: Re-scaled post-fix; brackets ~10 / 25 / 50 % of the data loss. See module docstring.
GRAPH_REG_LAMBDAS = [0.0, 2.6e-5, 6.5e-5, 1.3e-4]


def objective(trial: optuna.Trial) -> float:
    node_emb = trial.suggest_categorical("node_embeddings", NODE_EMBEDDINGS)
    # Swept rather than pinned so the sweep re-derives, rather than assumes, that a free
    # identity table hurts: with a strain-level split a VAL gene's embedding never receives a
    # gradient as a perturbed gene, so it can only memorize (observed train 0.86 / val 0.02).
    learnable = trial.suggest_categorical("learnable_embedding", [True, False])

    dist = trial.suggest_categorical("dist", ["point", "crps", "quantile"])

    hidden = trial.suggest_categorical("hidden_channels", [90, 180])
    layers = trial.suggest_categorical("num_transformer_layers", [2, 4, 6])
    graph_reg = trial.suggest_categorical("graph_reg_lambda", GRAPH_REG_LAMBDAS)
    profile_name = trial.suggest_categorical("hp_profile", list(PROFILES))
    profile = PROFILES[profile_name]
    target_norm = trial.suggest_categorical("target_norm", ["zscore", "yeo_johnson"])

    normalize_list = [] if target_norm == "zscore" else [ARM]
    standardize_list = [ARM] if target_norm == "zscore" else []

    overrides = [
        f"multitask.active_heads=[{ARM}]",
        f"cell_dataset.node_embeddings=[{node_emb}]",
        f"model.learnable_embedding.enabled={learnable}",
        f"multitask.decoder={DECODER}",
        f"multitask.dist={dist}",
        f"multitask.normalize_vector_targets=[{','.join(normalize_list)}]",
        f"multitask.standardize_per_feature_target=[{','.join(standardize_list)}]",
        f"model.hidden_channels={hidden}",
        f"model.num_transformer_layers={layers}",
        f"model.dropout={profile['dropout']}",
        f"model.graph_regularization.graph_reg_lambda={graph_reg}",
        f"regression_task.optimizer.lr={profile['lr']}",
        f"regression_task.optimizer.weight_decay={profile['weight_decay']}",
        f"data_module.num_workers={os.getenv('NUM_WORKERS', '4')}",
        f"wandb.project=torchcell_020_{ARM}",
        f"wandb.tags=[020,optuna,{ARM},{node_emb},{dist},"
        f"trial-{trial.number},{profile_name}]",
    ]

    with initialize_config_dir(version_base=None, config_dir=CONF_DIR):
        cfg = compose(config_name=BASE_CONFIG, overrides=overrides)

    print(
        f"[{ARM} w{WORKER_ID}] trial {trial.number}: emb={node_emb} learnable={learnable} "
        f"dist={dist} hidden={hidden} layers={layers} "
        f"graph_reg={graph_reg} profile={profile_name} norm={target_norm}",
        flush=True,
    )
    print(OmegaConf.to_yaml(cfg.multitask), flush=True)

    metrics = run_training(cfg)
    torch.cuda.empty_cache()

    # Record BOTH correlations regardless of which one ranks, so the study can be re-read on
    # the other metric without re-running -- and so a Pearson/Spearman divergence (the tell
    # for a scale artifact on the ordinal arm) is visible after the fact.
    for name, key in (
        ("pearson", f"val/{ARM}/pearson_per_feature_max"),
        ("spearman", f"val/{ARM}/spearman_per_feature_max"),
        ("pearson_per_instance", f"val/{ARM}/pearson_per_instance_max"),
    ):
        trial.set_user_attr(name, metrics.get(key))
    # The mean-collapse diagnostic: per-instance stays high under collapse, so a large gap
    # vs per-feature is the signature. Also record the graph-prior strength actually reached.
    trial.set_user_attr("val_loss_last", metrics.get("val/loss"))
    trial.set_user_attr("graph_reg_ratio", metrics.get("val/graph_reg/ratio_to_data"))
    trial.set_user_attr("node_embeddings", node_emb)
    trial.set_user_attr("learnable_embedding", learnable)
    trial.set_user_attr("dist", dist)

    if OBJECTIVE_METRIC not in metrics:
        raise optuna.TrialPruned(
            f"{OBJECTIVE_METRIC} not logged (keys: {sorted(metrics)[:12]}...)"
        )
    return float(metrics[OBJECTIVE_METRIC])


def get_study() -> optuna.Study:
    """Create-or-load. Single-objective maximize -- each arm has exactly one honest metric."""
    sampler = optuna.samplers.TPESampler(seed=WORKER_ID, multivariate=True, group=True)
    return optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE,
        sampler=sampler,
        load_if_exists=True,
        direction="maximize",
    )


def main() -> None:
    study = get_study()
    # --create-only: slurm runs this ONCE before the worker so a fresh SQLite file's DDL is
    # not raced by concurrent workers.
    if "--create-only" in sys.argv:
        print(f"[create-only] study={STUDY_NAME} direction={study.direction}", flush=True)
        return

    print(
        f"[{ARM} w{WORKER_ID}] study={STUDY_NAME} objective={OBJECTIVE_METRIC} "
        f"n_trials={N_TRIALS} base={BASE_CONFIG}",
        flush=True,
    )
    study.optimize(objective, n_trials=N_TRIALS, catch=(Exception,))

    if WORKER_ID == 0 and study.best_trial is not None:
        print(f"[{ARM} w0] best={study.best_value:.4f} params={study.best_params}")


if __name__ == "__main__":
    import multiprocessing as mp

    mp.set_start_method("spawn", force=True)
    main()
