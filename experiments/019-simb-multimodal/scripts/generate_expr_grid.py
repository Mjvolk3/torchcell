# experiments/019-simb-multimodal/scripts/generate_expr_grid.py
# [[experiments.019-simb-multimodal.scripts.generate_expr_grid]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/generate_expr_grid
"""Generate the single-GPU Fig-3 EXPRESSION sniff-sweep grid (SIMB 2026).

Writes ``gh_expr_grid_<NNN>.yaml`` configs + ``grid_manifest.json`` for the overnight
job-array launched by ``gh_cgt_multitask_array.slurm``. Each config is a SMALL-model,
single-GPU (``devices=1``, ``strategy=auto``, plain ``python``, NO DDP) expression-only
(``active_heads=[per_gene]``) run, capped at ``max_epochs=100`` + EarlyStopping(patience=20)
so it finishes in minutes-to-~1h. The ``%4`` array throttle keeps exactly 4 GPUs busy; the
grid is over-provisioned so the queue fills the ~16h overnight window (~64 GPU-hours) — if it
drains early, that is simply more replicate data.

Design (~16h / ~64 GPU-hour SNIFF sweep — separate real expression signal from noise):

PRIMARY levers, FULLY crossed (5 factors, 48 combos):
  * hidden_channels        {16, 32, 64}
  * num_transformer_layers {2, 3}
  * target                 raw  vs  per-gene-standardized (z-score each gene on TRAIN)
  * dataset                Kemmeren-only  vs  Kemmeren+Sameith
  * graph_reg_lambda       {0.0, 0.001}

SECONDARY hyperparameter profiles (2, bundled — a screen, not a clean factorial):
  * baseline   : lr=3e-4, dropout=0.1, weight_decay=1e-8
  * aggressive : lr=1e-3, dropout=0.0, weight_decay=1e-4

SEEDS: 3 per (primary × secondary) combo, so real signal is separable from split/init noise.

Total = 48 primary × 2 profiles × 3 seeds = 288 runs (array 0-287%4).

Priority ordering (per the brief): model size, target-standardization, dataset are primary
and fully crossed; lr/dropout/weight_decay are bundled into 2 profiles to cap the count.
Every knob + seed is written to ``grid_manifest.json`` (index -> knobs) so results are
analyzable after the fact.

Dataset restriction (Kemmeren-only vs +Sameith) — the fig3_core build fuses Kemmeren +
Sameith(Sm/Dm) expression, all under the SAME phenotype type ``expression_log2_ratio``, so
there is no per-head phenotype switch. We restrict at the ROW level via the config key
``cell_dataset.restrict_dataset_names`` (consumed in ``train_cgt_multitask.run_training``),
which intersects each split with ``dataset.dataset_name_index``. Pure Kemmeren is the exact
key ``MicroarrayKemmeren2014Dataset``; the mean-merged
``MicroarrayKemmeren2014Dataset+SmMicroarraySameith2015Dataset`` twin is a SEPARATE key that
is EXCLUDED from Kemmeren-only (that cross-platform mean-merge is a confound) and INCLUDED in
+Sameith along with ``DmMicroarraySameith2015Dataset``.

Grid configs disable the warmup-restarts LR scheduler (``regression_task.lr_scheduler=null``)
because ``CosineAnnealingWarmupRestarts`` drives the effective LR from ``max_lr`` and would
override the swept ``optimizer.lr``; with it off the ``lr`` knob is honest.
"""

import itertools
import json
import os
import os.path as osp
from typing import Any

import yaml

HERE = osp.dirname(osp.abspath(__file__))
CONF_DIR = osp.normpath(osp.join(HERE, "../conf"))
RESULTS_DIR = osp.normpath(osp.join(HERE, "../results"))

# Dataset-name restriction sets (exact keys in the fig3_core dataset_name_index).
KEMMEREN_ONLY = ["MicroarrayKemmeren2014Dataset"]
KEMMEREN_SAMEITH = [
    "MicroarrayKemmeren2014Dataset",
    "MicroarrayKemmeren2014Dataset+SmMicroarraySameith2015Dataset",
    "DmMicroarraySameith2015Dataset",
]

# PRIMARY levers (fully crossed).
HIDDEN_CHANNELS = [16, 32, 64]
NUM_LAYERS = [2, 3]
TARGET_MODES = ["raw", "std"]  # std -> standardize_per_feature_target: [per_gene]
DATASET_MODES = ["kemmeren", "kem_sam"]
GRAPH_REG_LAMBDAS = [0.0, 0.001]

# SECONDARY hyperparameter profiles (bundled).
SECONDARY_PROFILES = {
    "baseline": {"learning_rate": 3.0e-4, "dropout": 0.1, "weight_decay": 1.0e-8},
    "aggressive": {"learning_rate": 1.0e-3, "dropout": 0.0, "weight_decay": 1.0e-4},
}

SEEDS = [0, 1, 2]

# Small models -> fixed attention-head count that divides every hidden_channels (16/32/64).
NUM_ATTENTION_HEADS = 4


def build_config(
    *,
    seed: int,
    hidden_channels: int,
    num_layers: int,
    target_mode: str,
    dataset_mode: str,
    graph_reg_lambda: float,
    profile_name: str,
    profile: dict[str, float],
) -> dict[str, Any]:
    """Assemble one grid config dict (inherits gh_cgt_multitask_expr_000, overrides knobs)."""
    restrict = KEMMEREN_ONLY if dataset_mode == "kemmeren" else KEMMEREN_SAMEITH
    standardize = ["per_gene"] if target_mode == "std" else []
    dropout = float(profile["dropout"])
    return {
        "defaults": ["gh_cgt_multitask_expr_000", "_self_"],
        "seed": seed,
        "dry_run": False,
        "wandb": {
            "project": "torchcell_019-simb-multimodal_cgt_multitask",
            "tags": ["ws-run", "gilahyper", "expr-grid", "single-gpu", dataset_mode],
        },
        "cell_dataset": {
            "graphs": ["physical", "regulatory"],
            "node_embeddings": [],
            "query_file": "fig3_core.cql",
            "dataset_tag": "fig3_core",
            "restrict_dataset_names": restrict,
        },
        "transforms": {"use_transforms": False},
        "data_module": {
            "is_perturbation_subset": False,
            "batch_size": 32,
            "num_workers": 8,
            "pin_memory": True,
            "prefetch": False,
        },
        "trainer": {
            "max_epochs": 100,
            "strategy": "auto",
            "accelerator": "gpu",
            "devices": 1,
            "overfit_batches": 0,
            "precision": "bf16-mixed",
            "log_every_n_steps": 10,
            "fast_dev_run": False,
            "early_stopping": {
                "enabled": True,
                "monitor": "val/loss",
                "mode": "min",
                "patience": 20,
                "min_delta": 0.0,
            },
        },
        "multitask": {
            "active_heads": ["per_gene"],
            "loss_fn": "mse",
            "normalize_vector_targets": [],
            "standardize_per_feature_target": standardize,
            "vector_norm_method": "yeo_johnson",
            "target_norm_eps": 1.0e-8,
            "drop_features": {},
            "degenerate_robust_cv": 0.01,
        },
        "model": {
            "hidden_channels": hidden_channels,
            "num_transformer_layers": num_layers,
            "num_attention_heads": NUM_ATTENTION_HEADS,
            "dropout": dropout,
            "learnable_embedding": {
                "enabled": True,
                "size": "${model.hidden_channels}",
                "preprocessor": {"num_layers": 2, "dropout": dropout},
            },
            "graph_regularization": {"graph_reg_lambda": graph_reg_lambda},
            "perturbation_head": {"num_heads": NUM_ATTENTION_HEADS, "dropout": dropout},
        },
        "regression_task": {
            "optimizer": {
                "type": "AdamW",
                "lr": float(profile["learning_rate"]),
                "weight_decay": float(profile["weight_decay"]),
            },
            # Disable warmup-restarts so optimizer.lr is the honest swept LR (see module doc).
            "lr_scheduler": None,
            "clip_grad_norm": True,
            "clip_grad_norm_max_norm": 10.0,
        },
    }


def main() -> None:
    os.makedirs(CONF_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    combos = list(
        itertools.product(
            HIDDEN_CHANNELS,
            NUM_LAYERS,
            TARGET_MODES,
            DATASET_MODES,
            GRAPH_REG_LAMBDAS,
            SECONDARY_PROFILES.items(),
            SEEDS,
        )
    )

    manifest: dict[str, Any] = {
        "description": (
            "SIMB 2026 Fig-3 expression single-GPU sniff sweep (~16h / ~64 GPU-hours). "
            "Expression-only per_gene head on fig3_core; small models; EarlyStopping; "
            "both pearson_per_gene and pearson_per_strain logged."
        ),
        "n_configs": len(combos),
        "array_throttle": 4,
        "primary_levers": {
            "hidden_channels": HIDDEN_CHANNELS,
            "num_transformer_layers": NUM_LAYERS,
            "target": TARGET_MODES,
            "dataset": DATASET_MODES,
            "graph_reg_lambda": GRAPH_REG_LAMBDAS,
        },
        "secondary_profiles": SECONDARY_PROFILES,
        "seeds": SEEDS,
        "dataset_restrictions": {
            "kemmeren": KEMMEREN_ONLY,
            "kem_sam": KEMMEREN_SAMEITH,
        },
        "runs": {},
    }

    written = 0
    for idx, (
        hidden,
        layers,
        target_mode,
        dataset_mode,
        graph_reg_lambda,
        (profile_name, profile),
        seed,
    ) in enumerate(combos):
        config_name = f"gh_expr_grid_{idx:03d}"
        cfg = build_config(
            seed=seed,
            hidden_channels=hidden,
            num_layers=layers,
            target_mode=target_mode,
            dataset_mode=dataset_mode,
            graph_reg_lambda=graph_reg_lambda,
            profile_name=profile_name,
            profile=profile,
        )
        header = (
            f"# experiments/019-simb-multimodal/conf/{config_name}.yaml\n"
            f"# AUTO-GENERATED by generate_expr_grid.py — DO NOT EDIT BY HAND.\n"
            f"# Fig-3 expression single-GPU sniff-sweep run {idx:03d}/{len(combos) - 1}.\n"
            f"# hidden={hidden} layers={layers} target={target_mode} "
            f"dataset={dataset_mode} graph_reg={graph_reg_lambda} "
            f"profile={profile_name} seed={seed}\n"
        )
        out_path = osp.join(CONF_DIR, f"{config_name}.yaml")
        with open(out_path, "w") as f:
            f.write(header)
            yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)
        written += 1

        manifest["runs"][str(idx)] = {
            "array_index": idx,
            "config_name": config_name,
            "seed": seed,
            "hidden_channels": hidden,
            "num_transformer_layers": layers,
            "num_attention_heads": NUM_ATTENTION_HEADS,
            "target": target_mode,
            "standardize_per_feature_target": (
                ["per_gene"] if target_mode == "std" else []
            ),
            "dataset": dataset_mode,
            "restrict_dataset_names": (
                KEMMEREN_ONLY if dataset_mode == "kemmeren" else KEMMEREN_SAMEITH
            ),
            "graph_reg_lambda": graph_reg_lambda,
            "secondary_profile": profile_name,
            "learning_rate": float(profile["learning_rate"]),
            "dropout": float(profile["dropout"]),
            "weight_decay": float(profile["weight_decay"]),
            "batch_size": 32,
            "max_epochs": 100,
            "early_stopping_patience": 20,
        }

    manifest_path = osp.join(RESULTS_DIR, "grid_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    n = len(combos)
    print(f"Wrote {written} configs to {CONF_DIR}")
    print(f"Wrote manifest ({n} runs) to {manifest_path}")
    print("Launch the array with:")
    print(
        f"  sbatch --array=0-{n - 1}%4 "
        "experiments/019-simb-multimodal/scripts/gh_cgt_multitask_array.slurm"
    )


if __name__ == "__main__":
    main()
