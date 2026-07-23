# experiments/019-simb-multimodal/scripts/optuna_joint_sweep.py
# [[experiments.019-simb-multimodal.scripts.optuna_joint_sweep]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/optuna_joint_sweep
"""Delta CONTROLLED expression<->morphology auxiliary-task Optuna sweep driver.

Answers: does expression data improve MORPHOLOGY prediction on Ohya (and does morphology
rescue expression's val floor)? The control is a FIXED instance set — the 1,440 genotypes
with BOTH modalities (`require_modalities: [expression_log2_ratio, calmorph]` in the base
config) — with only the active heads varied by CONDITION:

    CONDITION=expr   active_heads=[per_gene]          objective = val/per_gene/pearson_per_gene
    CONDITION=morph  active_heads=[global]            objective = val/global/pearson_per_gene
    CONDITION=joint  active_heads=[per_gene, global]  objective = mean(expr, morph) honest r

joint − morph = "does expression help Ohya morphology"; joint − expr = the reverse. Because
the instance set is identical across conditions, any difference is the auxiliary-task effect,
NOT a data-quantity confound.

One process = one Optuna worker (4 pinned per GPU by the Delta slurm), all on ONE shared
SQLite study PER CONDITION. Delta compute nodes have internet -> W&B ONLINE (no offline dance).

Environment (set by the Delta slurm):
    CONDITION           expr | morph | joint   (default: joint)
    OPTUNA_STORAGE      sqlite:////<scratch>/.../optuna_019_joint_<condition>.db   (required)
    OPTUNA_STUDY_NAME   default: joint_<condition>_000
    OPTUNA_N_TRIALS     trials THIS worker runs (default: 20)
    OPTUNA_WORKER_ID    0..3, seeds the sampler (default: 0)
    JOINT_BASE_CONFIG   base Hydra config (default: delta_joint_expr_morph_000)
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
BASE_CONFIG = os.getenv("JOINT_BASE_CONFIG", "delta_joint_expr_morph_000")
STORAGE = os.environ["OPTUNA_STORAGE"]
CONDITION = os.getenv("CONDITION", "joint")
STUDY_NAME = os.getenv("OPTUNA_STUDY_NAME", f"joint_{CONDITION}_000")
N_TRIALS = int(os.getenv("OPTUNA_N_TRIALS", "20"))
WORKER_ID = int(os.getenv("OPTUNA_WORKER_ID", "0"))

ACTIVE_HEADS = {
    "expr": ["per_gene"],
    "morph": ["global"],
    "joint": ["per_gene", "global"],
}[CONDITION]

EXPR_METRIC = "val/per_gene/pearson_per_gene"    # honest per-feature (across instances)
MORPH_METRIC = "val/global/pearson_per_gene"

PROFILES = {
    "baseline": {"lr": 3.0e-4, "dropout": 0.1, "weight_decay": 1.0e-8},
    "aggressive": {"lr": 1.0e-3, "dropout": 0.0, "weight_decay": 1.0e-4},
}


def _norm_choice(trial: optuna.Trial, head: str) -> str:
    """Per-modality target-norm lever. Expression: raw|zscore (zscore = anti-collapse);
    morphology: raw|yeo_johnson|zscore.
    """
    if head == "per_gene":
        return trial.suggest_categorical("expr_norm", ["raw", "zscore"])
    return trial.suggest_categorical("morph_norm", ["raw", "yeo_johnson", "zscore"])


def objective(trial: optuna.Trial) -> float | tuple[float, float]:
    hidden = trial.suggest_categorical("hidden_channels", [16, 32, 64])
    layers = trial.suggest_categorical("num_transformer_layers", [2, 3])
    graph_reg = trial.suggest_categorical("graph_reg_lambda", [0.0, 0.001])
    profile_name = trial.suggest_categorical("hp_profile", ["baseline", "aggressive"])
    profile = PROFILES[profile_name]

    # Build the two per-head normalization lists from each active head's sampled lever.
    normalize_list: list[str] = []   # -> Yeo-Johnson (vector_norm_method)
    standardize_list: list[str] = []  # -> per-feature z-score (anti-mean-collapse)
    for head in ACTIVE_HEADS:
        choice = _norm_choice(trial, head)
        if choice == "zscore":
            standardize_list.append(head)
        elif choice == "yeo_johnson":
            normalize_list.append(head)
        # "raw" -> neither list

    overrides = [
        f"multitask.active_heads=[{','.join(ACTIVE_HEADS)}]",
        f"multitask.normalize_vector_targets=[{','.join(normalize_list)}]",
        f"multitask.standardize_per_feature_target=[{','.join(standardize_list)}]",
        f"model.hidden_channels={hidden}",
        f"model.num_transformer_layers={layers}",
        f"model.dropout={profile['dropout']}",
        f"model.graph_regularization.graph_reg_lambda={graph_reg}",
        f"regression_task.optimizer.lr={profile['lr']}",
        f"regression_task.optimizer.weight_decay={profile['weight_decay']}",
        f"data_module.num_workers={os.getenv('NUM_WORKERS', '4')}",
        "wandb.tags=[ws-run,delta,joint,optuna,single-gpu,"
        f"{CONDITION},trial-{trial.number},{profile_name}]",
    ]

    with initialize_config_dir(version_base=None, config_dir=CONF_DIR):
        cfg = compose(config_name=BASE_CONFIG, overrides=overrides)

    print(
        f"[{CONDITION} w{WORKER_ID}] trial {trial.number}: heads={ACTIVE_HEADS} "
        f"hidden={hidden} layers={layers} graph_reg={graph_reg} profile={profile_name} "
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

    if CONDITION == "joint":
        # MULTI-OBJECTIVE: maximize BOTH honest metrics -> Optuna returns the Pareto front of
        # (expression, morphology). This is the scientific object: the configs on the frontier
        # of "good at both", NOT a hand-weighted scalar that hides the trade-off.
        if expr_r is None or morph_r is None:
            raise optuna.TrialPruned(
                f"joint needs both metrics (expr={expr_r}, morph={morph_r})"
            )
        return expr_r, morph_r

    objective_metric = EXPR_METRIC if CONDITION == "expr" else MORPH_METRIC
    if objective_metric not in metrics:
        raise optuna.TrialPruned(
            f"{objective_metric} not logged (keys: {sorted(metrics)[:12]}...)"
        )
    return metrics[objective_metric]


def get_study() -> optuna.Study:
    """Create-or-load the study. joint = MULTI-objective (maximize expr, maximize morph);
    expr/morph = single-objective. TPESampler handles both (MOTPE for the multi case)."""
    sampler = optuna.samplers.TPESampler(seed=WORKER_ID, multivariate=True, group=True)
    common = dict(
        study_name=STUDY_NAME, storage=STORAGE, sampler=sampler, load_if_exists=True
    )
    if CONDITION == "joint":
        return optuna.create_study(directions=["maximize", "maximize"], **common)
    return optuna.create_study(direction="maximize", **common)


def main() -> None:
    study = get_study()
    # --create-only: the slurm runs this ONCE (serialized) before the 4 workers so they only
    # load_if_exists — avoids the fresh-DB DDL race. The directions logic lives HERE (one place)
    # so pre-create and workers always agree on single- vs multi-objective.
    if "--create-only" in sys.argv:
        print(f"[create-only] study={STUDY_NAME} directions={study.directions}", flush=True)
        return

    print(
        f"[{CONDITION} w{WORKER_ID}] study={STUDY_NAME} heads={ACTIVE_HEADS} "
        f"n_trials={N_TRIALS} multi_obj={CONDITION == 'joint'}",
        flush=True,
    )
    study.optimize(objective, n_trials=N_TRIALS, catch=(Exception,))

    if WORKER_ID == 0:
        if CONDITION == "joint":
            print(f"[joint w0] Pareto front ({len(study.best_trials)} trials):")
            for t in study.best_trials[:10]:
                print(f"  t{t.number} (expr,morph)={[round(v, 4) for v in t.values]} {t.params}")
        else:
            print(f"[{CONDITION} w0] best={study.best_value:.4f} params={study.best_params}")


if __name__ == "__main__":
    import multiprocessing as mp

    mp.set_start_method("spawn", force=True)
    main()
