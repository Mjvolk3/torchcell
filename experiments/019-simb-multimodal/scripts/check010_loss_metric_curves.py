# experiments/019-simb-multimodal/scripts/check010_loss_metric_curves.py
# [[experiments.019-simb-multimodal.scripts.check010_loss_metric_curves]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/check010_loss_metric_curves
"""Fact-check: does the 010 task really show monotone losses with the val minimum last?

The retrospective (notes-tex/019-simb-multimodal/sections/7-common.tex) asserts that the
010 fitness-and-interaction task shows neither early overfitting nor the expression
loss/metric divergence, "with both losses falling monotonically and the validation-loss
minimum at the last epoch". Nothing in the 019 leaderboard cache covers the 010 project,
so this pulls the per-epoch history directly and measures it.

Per training run it records: total epochs, the epoch of the val-loss minimum, how far that
sits from the end, the rise from min to final, the number of epoch-to-epoch increases in
each loss (monotonicity), and the epoch of the headline metric peak
(`val/gene_interaction/Pearson`) relative to the loss minimum.

Run from repo root (needs network; W&B login):
  PYTHONPATH=. python experiments/019-simb-multimodal/scripts/check010_loss_metric_curves.py
"""

from __future__ import annotations

import os
import os.path as osp
import signal

import numpy as np
import pandas as pd
import wandb
from dotenv import load_dotenv

ENTITY = "zhao-group"
PROJECT = "torchcell_010-kuzmin-tmi_equivariant_cell_graph_transformer"

KEYS = [
    "epoch",
    "val/loss",
    "train/loss",
    "val/gene_interaction/Pearson",
    "val/gene_interaction/MSE",
    "val/gene_interaction/RMSE",
    "train/gene_interaction/Pearson",
    "val/point_loss",
    "val/dist_loss",
    "val/graph_reg_loss",
    "train/point_loss",
]

MIN_EPOCHS = 5  # anything shorter is an eval/smoke run, not a training curve
HISTORY_TIMEOUT_S = 120
SAMPLES = 5000  # runs here top out at ~64 epochs, so this is the full curve


class _Timeout(Exception):
    pass


def _handler(signum, frame):  # noqa: ARG001
    raise _Timeout


def fetch_history(run, keys: list[str]):
    """One key at a time, merged on `epoch`.

    A single multi-key `run.history` request returns an EMPTY frame here, because train and
    val scalars land on different global steps and the sampled-history join keeps only rows
    where every requested key is present. Asking for `['epoch', 'val/loss', 'train/loss']`
    on run lzs9pcj3 returns (0, 0) while `['epoch', 'val/loss']` returns (64, 3). So each
    key is pulled on its own and outer-merged, and a key logged more than once per epoch is
    reduced to its per-epoch mean.
    """
    signal.signal(signal.SIGALRM, _handler)
    signal.alarm(HISTORY_TIMEOUT_S)
    try:
        merged = None
        for key in keys:
            if key == "epoch":
                continue
            frame = run.history(keys=["epoch", key], samples=SAMPLES)
            if frame is None or frame.empty or key not in frame:
                continue
            # W&B returns the literal string "NaN" for a non-finite scalar, which makes the
            # column object dtype and breaks the groupby aggregation.
            frame = frame[["epoch", key]].copy()
            frame["epoch"] = pd.to_numeric(frame["epoch"], errors="coerce")
            frame[key] = pd.to_numeric(frame[key], errors="coerce")
            frame = frame.dropna(subset=["epoch"])
            frame = frame.groupby("epoch", as_index=False)[key].mean()
            merged = frame if merged is None else merged.merge(frame, on="epoch", how="outer")
        return merged
    except _Timeout:
        return None
    finally:
        signal.alarm(0)


def _n_increases(values: np.ndarray) -> int:
    d = np.diff(values)
    return int((d > 0).sum())


def _n_decreases(values: np.ndarray) -> int:
    d = np.diff(values)
    return int((d < 0).sum())


def main() -> None:
    load_dotenv()
    out_dir = osp.join(
        os.environ.get("EXPERIMENT_ROOT", "experiments"),
        "019-simb-multimodal",
        "results",
    )
    if not osp.isdir(out_dir):
        out_dir = osp.join("experiments", "019-simb-multimodal", "results")
    os.makedirs(out_dir, exist_ok=True)

    api = wandb.Api(timeout=60)
    runs = list(api.runs(f"{ENTITY}/{PROJECT}"))
    print(f"{PROJECT}: {len(runs)} runs total")

    train_runs = []
    for r in runs:
        ep = r.summary.get("epoch", -1)
        try:
            ep = float(ep)
        except (TypeError, ValueError):
            ep = -1.0
        if ep >= MIN_EPOCHS and "val/loss" in r.summary:
            train_runs.append((ep, r))
    train_runs.sort(key=lambda t: t[0], reverse=True)
    print(f"training runs with >= {MIN_EPOCHS} epochs and a val/loss: {len(train_runs)}")

    rows = []
    for ep, run in train_runs:
        present = [k for k in KEYS if k == "epoch" or k in run.summary]
        hist = fetch_history(run, present)
        if hist is None or hist.empty:
            print(f"  {run.id}: history unavailable")
            continue
        hist = hist.sort_values("epoch").reset_index(drop=True)
        hist.to_csv(osp.join(out_dir, f"check010_curve_{run.id}.csv"), index=False)

        row: dict[str, object] = {
            "run_id": run.id,
            "run_name": run.name,
            "state": run.state,
            "created_at": str(run.created_at),
            "n_points": len(hist),
            "last_epoch": float(hist["epoch"].iloc[-1]),
            "max_epochs_cfg": run.config.get("trainer", {}).get("max_epochs"),
            "lr": run.config.get("regression_task", {})
            .get("optimizer", {})
            .get("lr"),
        }

        for name, key in (("val_loss", "val/loss"), ("train_loss", "train/loss")):
            if key not in hist:
                continue
            v = hist[key].to_numpy(dtype=float)
            e = hist["epoch"].to_numpy(dtype=float)
            ok = np.isfinite(v)
            v, ee = v[ok], e[ok]
            if v.size == 0:
                continue
            imin = int(np.argmin(v))
            row[f"{name}_min"] = float(v[imin])
            row[f"{name}_min_epoch"] = float(ee[imin])
            row[f"{name}_final"] = float(v[-1])
            row[f"{name}_epochs_after_min"] = float(ee[-1] - ee[imin])
            row[f"{name}_rise_min_to_final"] = float(v[-1] - v[imin])
            row[f"{name}_rise_frac"] = (
                float((v[-1] - v[imin]) / abs(v[imin])) if v[imin] != 0 else float("nan")
            )
            row[f"{name}_n_increases"] = _n_increases(v)
            row[f"{name}_n_steps"] = int(v.size - 1)
            row[f"{name}_monotone_decreasing"] = bool(_n_increases(v) == 0)

        for name, key in (
            ("val_gi_pearson", "val/gene_interaction/Pearson"),
            ("val_gi_mse", "val/gene_interaction/MSE"),
        ):
            if key not in hist:
                continue
            v = hist[key].to_numpy(dtype=float)
            e = hist["epoch"].to_numpy(dtype=float)
            ok = np.isfinite(v)
            v, ee = v[ok], e[ok]
            if v.size == 0:
                continue
            if "Pearson" in key:
                ibest = int(np.argmax(v))
                row[f"{name}_max"] = float(v[ibest])
            else:
                ibest = int(np.argmin(v))
                row[f"{name}_min"] = float(v[ibest])
            row[f"{name}_best_epoch"] = float(ee[ibest])
            row[f"{name}_final"] = float(v[-1])

        rows.append(row)
        print(
            f"  {run.id} epochs={row['last_epoch']:.0f} "
            f"val_loss_min@{row.get('val_loss_min_epoch')} "
            f"rises={row.get('val_loss_n_increases')}/{row.get('val_loss_n_steps')} "
            f"pearson_peak@{row.get('val_gi_pearson_best_epoch')}"
        )

    df = pd.DataFrame(rows)
    out = osp.join(out_dir, "check010_run_summary.csv")
    df.to_csv(out, index=False)
    print(f"\nwrote {out}  ({len(df)} rows)")
    with pd.option_context("display.width", 250, "display.max_columns", 60):
        cols = [
            c
            for c in [
                "run_id",
                "last_epoch",
                "val_loss_min_epoch",
                "val_loss_epochs_after_min",
                "val_loss_rise_min_to_final",
                "val_loss_n_increases",
                "val_loss_n_steps",
                "train_loss_min_epoch",
                "train_loss_n_increases",
                "val_gi_pearson_max",
                "val_gi_pearson_best_epoch",
                "val_gi_pearson_final",
            ]
            if c in df
        ]
        print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
