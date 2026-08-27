# experiments/019-simb-multimodal/scripts/pull_round_leaderboards.py
# [[experiments.019-simb-multimodal.scripts.pull_round_leaderboards]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/pull_round_leaderboards
"""One leaderboard across every phenotype strand run for the SIMB multimodal figure.

WHY THIS EXISTS. The six strands (expression, morphology, joint expression+morphology,
betaxanthin, beta-carotene, amino-acid metabolome, and the joint betaxanthin+metabolome
replication) each live in their own W&B project, were scored at different times, and are
quoted in prose from whichever run someone happened to be looking at. A retrospective that
compares them has to read them all the same way or it is comparing scoring conventions
rather than strands. This dumps one CSV, from which every retrospective number is read.

THE SCORING RULE, and it is named because it is biased. For each run this records:

  last      the metric at the final logged epoch. What `run.summary` holds, and what the
            expression round proved can sit ~1,300 epochs before the good checkpoint.
  roll_max  the maximum of a centered 5-epoch rolling mean of the validation metric. An
            UPWARD-BIASED order statistic whose bias grows with the number of epochs run,
            so it is only comparable between runs of similar length; `epochs` is carried
            alongside for exactly that reason. Same rule as
            `experiments/019-simb-multimodal/scripts/wave4b_convergence.py`.

Neither is "the score". Both are reported so a reader can see when they disagree, which is
itself one of the round's findings.

TRUNCATION IS RECORDED, NOT HIDDEN. `epochs` is the last logged epoch and `state` is W&B's,
so a run that was walltime-killed while still climbing is visible as such. Runs whose
metric is identically zero are collapsed predictors (the lr = 1e-3 cells), flagged by
`is_collapsed` rather than dropped, because dropping them would silently improve every
per-project maximum.

W&B history is DOWNSAMPLED to at most `samples` points per run, so `epoch_at_roll_max` is
approximate to that resolution. It is a locator, not a measurement.

Run from repo root (needs network; W&B login):
  PYTHONPATH=. python experiments/019-simb-multimodal/scripts/pull_round_leaderboards.py
"""

from __future__ import annotations

import json
import os
import os.path as osp

import numpy as np
import pandas as pd
import wandb
from dotenv import load_dotenv

ENTITY = "zhao-group"
EXPERIMENT = "019-simb-multimodal"

# (strand, project, primary metric, extra metrics). The primary metric is the one the
# strand's objective maximized; extras are carried so a strand can be re-read on a second
# axis without a second pull. Beta-carotene's primary is SPEARMAN because its target is a
# subjective ordinal colony-color score, for which a Pearson ceiling is the wrong object
# (see pigment_noise_ceiling.py).
STRANDS: list[tuple[str, str, str, list[str]]] = [
    ("expression", "torchcell_019_expr_v8", "val/expression/pearson_per_feature", []),
    ("expression_masked", "torchcell_019_expr_v9", "val/expression/pearson_per_feature", []),
    ("morphology", "torchcell_019_morph_v5", "val/morphology/pearson_per_feature", []),
    (
        "expression_morphology_joint",
        "torchcell_019_expr_morph_v5",
        "val/mean/pearson_per_feature",
        [
            "val/expression/pearson_per_feature",
            "val/morphology/pearson_per_feature",
        ],
    ),
    (
        "betaxanthin",
        "torchcell_020_betaxanthin_v4",
        "val/betaxanthin/pearson_per_feature",
        ["val/betaxanthin/spearman_per_feature"],
    ),
    (
        "beta_carotene",
        "torchcell_021_beta_carotene_v4",
        "val/beta_carotene/spearman_per_feature",
        ["val/beta_carotene/pearson_per_feature"],
    ),
    (
        "amino_acid",
        "torchcell_022_mulleder19_v4",
        "val/mulleder19/pearson_per_feature",
        ["val/mulleder19/spearman_per_feature"],
    ),
    (
        "betaxanthin_amino_acid_joint",
        "torchcell_023_bx_m19_v1",
        "val/betaxanthin/pearson_per_feature",
        [
            "val/betaxanthin/spearman_per_feature",
            "val/mulleder19/pearson_per_feature",
        ],
    ),
]

# Config keys worth carrying into the leaderboard. Flat keys only; the nested Hydra tree is
# not reconstructed here because the training harness already flattens what varies into
# W&B config at launch.
CONFIG_KEYS = [
    "seed",
    "lr",
    "dropout",
    "num_layers",
    "hidden_channels",
    "target_norm",
    "graph_prior",
    "dist",
    "decoder",
    "n_train_supervised",
    "n_val_supervised",
    "n_test_supervised",
    "total_param_count",
    "perf/epoch_seconds",
]

ROLL_WINDOW = 5
HISTORY_SAMPLES = 2000


def _roll_max(values: np.ndarray, window: int = ROLL_WINDOW) -> tuple[float, int]:
    """Max of a centered rolling mean, and the index it occurs at."""
    finite = np.isfinite(values)
    if finite.sum() == 0:
        return float("nan"), -1
    series = pd.Series(values).rolling(window, center=True, min_periods=1).mean()
    idx = int(series.idxmax())
    return float(series.iloc[idx]), idx


def pull(api: wandb.Api, strand: str, project: str, primary: str, extras: list[str]):
    rows = []
    keys = [primary, *extras]
    for run in api.runs(f"{ENTITY}/{project}"):
        row: dict[str, object] = {
            "strand": strand,
            "project": project,
            "run_id": run.id,
            "run_name": run.name,
            "state": run.state,
            "tags": ",".join(run.tags),
            "primary_metric": primary,
        }
        for key in CONFIG_KEYS:
            row[key.replace("/", "_")] = run.config.get(key, run.summary.get(key))
        # ONE REQUEST PER KEY, deliberately. `run.history(keys=[a, b])` keeps only the
        # rows where BOTH a and b are present, so asking for the primary metric together
        # with an auxiliary head's metric returns an EMPTY frame for every run that does
        # not carry that head. In a paired design that is exactly the control arm, so the
        # joined form silently drops one side of every comparison the round exists to
        # make.
        row["epochs"] = None
        row["n_history_points"] = 0
        for key in keys:
            history = run.history(keys=["epoch", key], samples=HISTORY_SAMPLES)
            if history.empty or key not in history:
                continue
            epochs = (
                history["epoch"].to_numpy()
                if "epoch" in history
                else np.arange(len(history))
            )
            values = history[key].to_numpy(dtype=float)
            best, idx = _roll_max(values)
            tag = (
                "primary"
                if key == primary
                else key.split("/")[-2] + "_" + key.split("/")[-1]
            )
            finite = np.isfinite(values)
            row[f"{tag}_last"] = float(values[finite][-1]) if finite.any() else None
            row[f"{tag}_roll_max"] = best
            row[f"{tag}_epoch_at_roll_max"] = (
                float(epochs[idx]) if 0 <= idx < len(epochs) else None
            )
            if key == primary:
                row["epochs"] = float(np.nanmax(epochs))
                row["n_history_points"] = int(len(history))
                # A run whose validation metric never leaves zero is a collapsed
                # constant predictor, not a weak model. Flagged, never dropped.
                row["is_collapsed"] = bool(np.nanmax(np.abs(values)) < 1e-6)
        rows.append(row)
    return rows


def summarize(df: pd.DataFrame) -> dict[str, object]:
    out: dict[str, object] = {}
    for strand, group in df.groupby("strand"):
        live = group[~group["is_collapsed"].fillna(False)]
        best = live.loc[live["primary_roll_max"].idxmax()] if len(live) else None
        out[str(strand)] = {
            "project": str(group["project"].iloc[0]),
            "metric": str(group["primary_metric"].iloc[0]),
            "n_runs": int(len(group)),
            "n_collapsed": int(group["is_collapsed"].fillna(False).sum()),
            "epochs_min": float(group["epochs"].min()) if group["epochs"].notna().any() else None,
            "epochs_max": float(group["epochs"].max()) if group["epochs"].notna().any() else None,
            "best_roll_max": float(best["primary_roll_max"]) if best is not None else None,
            "best_run_id": str(best["run_id"]) if best is not None else None,
            "best_run_epochs": float(best["epochs"]) if best is not None else None,
            "best_epoch_at_roll_max": (
                float(best["primary_epoch_at_roll_max"]) if best is not None else None
            ),
            "median_roll_max": (
                float(live["primary_roll_max"].median()) if len(live) else None
            ),
        }
    return out


def main() -> None:
    load_dotenv()
    experiment_root = os.environ["EXPERIMENT_ROOT"]
    results_dir = osp.join(experiment_root, EXPERIMENT, "results")
    os.makedirs(results_dir, exist_ok=True)

    api = wandb.Api(timeout=60)
    rows: list[dict[str, object]] = []
    for strand, project, primary, extras in STRANDS:
        print(f"pulling {strand} <- {project}", flush=True)
        rows.extend(pull(api, strand, project, primary, extras))
    df = pd.DataFrame(rows)
    csv_path = osp.join(results_dir, "round_leaderboards.csv")
    df.to_csv(csv_path, index=False)

    summary = summarize(df)
    with open(osp.join(results_dir, "round_leaderboards_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"-> {csv_path}")


if __name__ == "__main__":
    main()
