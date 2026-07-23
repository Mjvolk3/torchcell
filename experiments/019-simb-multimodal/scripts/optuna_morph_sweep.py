# experiments/019-simb-multimodal/scripts/optuna_morph_sweep.py
# [[experiments.019-simb-multimodal.scripts.optuna_morph_sweep]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/optuna_morph_sweep
"""Optuna MORPHOLOGY sniff sweep driver (IGB mmli, OFFLINE, single-GPU per trial).

One process = one Optuna WORKER. The IGB `mmli` slurm launcher starts 4 of these (one per
GPU, pinned with ``CUDA_VISIBLE_DEVICES``), all pointing at the SAME SQLite study on shared
scratch, so trials coordinate through the DB — Optuna's canonical multi-worker pattern. This
fits a SINGLE-node partition far better than Hydra's submitit launcher (which submits many
child SLURM jobs that would just queue behind each other on one node).

Each trial samples the same small-model levers as the GilaHyper expression grid, but on the
CalMorph MORPHOLOGY `global` head (278 features, n_train=3757), and MAXIMIZES the honest
per-feature metric ``val/global/pearson_per_gene`` (per-feature Pearson across instances —
NOT the per-strain metric, which stays high under mean-collapse).

Search space (fixed split seed=0 so the objective is comparable across trials; seed
replication of the WINNERS is a separate follow-up, per the experimental-plans note):
    hidden_channels        {16, 32, 64}
    num_transformer_layers {2, 3}
    target_norm            {raw, yeo_johnson, zscore}   # zscore = anti-mean-collapse lever
    graph_reg_lambda       {0.0, 0.001}
    hp_profile             {baseline, aggressive}       # bundled lr/dropout/weight_decay
  -> 3*2*3*2*2 = 72 discrete combos.

Environment (set by the slurm launcher):
    OPTUNA_STORAGE      sqlite:////<scratch>/.../optuna_019_morph.db   (required)
    OPTUNA_STUDY_NAME   study name (default: morph_000)
    OPTUNA_N_TRIALS     trials THIS worker runs (default: 20)
    OPTUNA_WORKER_ID    0..3, seeds the sampler for cross-worker diversity (default: 0)
    MORPH_BASE_CONFIG   base Hydra config name (default: igb_mmli_optuna_morph_000)
    WANDB_MODE          offline (set by launcher; consumed inside run_training)

Run standalone (from repo root) for a 1-GPU smoke test:
    OPTUNA_STORAGE=sqlite:////tmp/optuna_morph_smoke.db OPTUNA_N_TRIALS=1 \
      python experiments/019-simb-multimodal/scripts/optuna_morph_sweep.py
"""

import os
import os.path as osp

import optuna
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from train_cgt_multitask import run_training

CONF_DIR = osp.abspath(osp.join(osp.dirname(__file__), "../conf"))
BASE_CONFIG = os.getenv("MORPH_BASE_CONFIG", "igb_mmli_optuna_morph_000")
STORAGE = os.environ["OPTUNA_STORAGE"]
STUDY_NAME = os.getenv("OPTUNA_STUDY_NAME", "morph_000")
N_TRIALS = int(os.getenv("OPTUNA_N_TRIALS", "20"))
WORKER_ID = int(os.getenv("OPTUNA_WORKER_ID", "0"))

OBJECTIVE_METRIC = "val/global/pearson_per_gene"  # honest per-feature (across-instances)

# Bundled secondary hyperparameter profiles (mirror the expression grid).
PROFILES = {
    "baseline": {"lr": 3.0e-4, "dropout": 0.1, "weight_decay": 1.0e-8},
    "aggressive": {"lr": 1.0e-3, "dropout": 0.0, "weight_decay": 1.0e-4},
}


def _target_norm_overrides(target_norm: str) -> list[str]:
    """Hydra overrides selecting the target-normalization lever for the `global` head."""
    if target_norm == "raw":
        return [
            "multitask.normalize_vector_targets=[]",
            "multitask.standardize_per_feature_target=[]",
        ]
    if target_norm == "yeo_johnson":
        return [
            "multitask.normalize_vector_targets=[global]",
            "multitask.standardize_per_feature_target=[]",
            "multitask.vector_norm_method=yeo_johnson",
        ]
    if target_norm == "zscore":
        # Anti-mean-collapse lever: per-feature z-score forces prediction of DEVIATIONS.
        return [
            "multitask.normalize_vector_targets=[]",
            "multitask.standardize_per_feature_target=[global]",
        ]
    raise ValueError(f"unknown target_norm {target_norm!r}")


def objective(trial: optuna.Trial) -> float:
    hidden = trial.suggest_categorical("hidden_channels", [16, 32, 64])
    layers = trial.suggest_categorical("num_transformer_layers", [2, 3])
    target_norm = trial.suggest_categorical(
        "target_norm", ["raw", "yeo_johnson", "zscore"]
    )
    graph_reg = trial.suggest_categorical("graph_reg_lambda", [0.0, 0.001])
    profile_name = trial.suggest_categorical("hp_profile", ["baseline", "aggressive"])
    profile = PROFILES[profile_name]

    overrides = [
        f"model.hidden_channels={hidden}",
        f"model.num_transformer_layers={layers}",
        f"model.dropout={profile['dropout']}",
        f"model.graph_regularization.graph_reg_lambda={graph_reg}",
        f"regression_task.optimizer.lr={profile['lr']}",
        f"regression_task.optimizer.weight_decay={profile['weight_decay']}",
        # 4 workers share the node CPUs, so each trial gets fewer dataloader workers than
        # the base config's default (launcher exports NUM_WORKERS = cpus-per-task / 4).
        f"data_module.num_workers={os.getenv('NUM_WORKERS', '4')}",
        # Tag the offline W&B run so it can be joined to this trial after `wandb sync`.
        "wandb.tags=[ws-run,igb,mmli,morph,optuna,single-gpu,"
        f"trial-{trial.number},{target_norm},{profile_name}]",
    ]
    overrides.extend(_target_norm_overrides(target_norm))

    with initialize_config_dir(version_base=None, config_dir=CONF_DIR):
        cfg = compose(config_name=BASE_CONFIG, overrides=overrides)

    print(
        f"[worker {WORKER_ID}] trial {trial.number}: hidden={hidden} layers={layers} "
        f"target_norm={target_norm} graph_reg={graph_reg} profile={profile_name}",
        flush=True,
    )
    print(OmegaConf.to_yaml(cfg.multitask), flush=True)

    metrics = run_training(cfg)
    torch.cuda.empty_cache()

    if OBJECTIVE_METRIC not in metrics:
        raise optuna.TrialPruned(
            f"{OBJECTIVE_METRIC} not logged (keys: {sorted(metrics)[:12]}...)"
        )
    value = metrics[OBJECTIVE_METRIC]
    trial.set_user_attr("val_pearson_per_strain", metrics.get("val/global/pearson_per_strain"))
    trial.set_user_attr("val_loss", metrics.get("val/loss"))
    return value


def main() -> None:
    sampler = optuna.samplers.TPESampler(
        seed=WORKER_ID, multivariate=True, group=True
    )
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE,
        direction="maximize",
        sampler=sampler,
        load_if_exists=True,
    )
    print(
        f"[worker {WORKER_ID}] study={STUDY_NAME} storage={STORAGE} "
        f"n_trials={N_TRIALS} objective=maximize({OBJECTIVE_METRIC})",
        flush=True,
    )
    # catch=() would kill the worker on the first bad trial; catch Exception so one failed
    # trial is marked FAILED and the worker keeps pulling from the shared study.
    study.optimize(objective, n_trials=N_TRIALS, catch=(Exception,))

    if WORKER_ID == 0:
        print(f"[worker 0] best value={study.best_value:.4f} params={study.best_params}")


if __name__ == "__main__":
    import multiprocessing as mp

    mp.set_start_method("spawn", force=True)
    main()
