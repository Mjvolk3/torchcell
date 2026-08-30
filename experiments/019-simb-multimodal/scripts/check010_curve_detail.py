# experiments/019-simb-multimodal/scripts/check010_curve_detail.py
# [[experiments.019-simb-multimodal.scripts.check010_curve_detail]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/check010_curve_detail
"""Detail pass on the 010 curves dumped by `check010_loss_metric_curves.py`.

Answers three things the summary table cannot: how LARGE the non-monotone rises in the two
losses are (a rise of 0.001% is not a finding, a rise of 30% is), where the headline
correlation peaks under the same 5-epoch centered rolling mean the 019 leaderboard uses
(a raw argmax over 64 noisy epochs is an upward-biased order statistic), and how far that
smoothed peak sits from the epoch the validation loss bottoms.

Reads only the local `results/check010_curve_<id>.csv` dumps; no network.

  PYTHONPATH=. python experiments/019-simb-multimodal/scripts/check010_curve_detail.py
"""

from __future__ import annotations

import glob
import os
import os.path as osp

import numpy as np
import pandas as pd
from dotenv import load_dotenv

ROLL_WINDOW = 5


def _rises(v: np.ndarray) -> tuple[int, float, float]:
    """Count of increases, largest single relative increase, and the total increase mass."""
    d = np.diff(v)
    up = d > 0
    if not up.any():
        return 0, 0.0, 0.0
    rel = d[up] / np.abs(v[:-1][up])
    return int(up.sum()), float(np.nanmax(rel)), float(d[up].sum())


def main() -> None:
    load_dotenv()
    res = osp.join(
        os.environ.get("EXPERIMENT_ROOT", "experiments"), "019-simb-multimodal", "results"
    )
    if not osp.isdir(res):
        res = osp.join("experiments", "019-simb-multimodal", "results")

    summary = pd.read_csv(osp.join(res, "check010_run_summary.csv"))
    summary = summary.sort_values("val_gi_pearson_max", ascending=False)

    rows = []
    for _, s in summary.iterrows():
        rid = s["run_id"]
        path = osp.join(res, f"check010_curve_{rid}.csv")
        if not osp.exists(path):
            continue
        h = pd.read_csv(path).sort_values("epoch").reset_index(drop=True)
        row: dict[str, object] = {"run_id": rid, "last_epoch": h["epoch"].max()}

        for name, key in (("val_loss", "val/loss"), ("train_loss", "train/loss")):
            if key not in h:
                continue
            m = np.isfinite(h[key].to_numpy(dtype=float))
            v = h[key].to_numpy(dtype=float)[m]
            e = h["epoch"].to_numpy(dtype=float)[m]
            n_up, max_rel, mass = _rises(v)
            row[f"{name}_n_up"] = n_up
            row[f"{name}_n_steps"] = int(v.size - 1)
            row[f"{name}_max_rel_rise"] = max_rel
            row[f"{name}_min_epoch"] = float(e[int(np.argmin(v))])
            row[f"{name}_first"] = float(v[0])
            row[f"{name}_min"] = float(np.min(v))
            row[f"{name}_final"] = float(v[-1])
            # Same, restricted to the second half, where the curve is no longer dominated
            # by the collapse out of the initialization.
            half = v.size // 2
            n_up_h, max_rel_h, _ = _rises(v[half:])
            row[f"{name}_n_up_2ndhalf"] = n_up_h
            row[f"{name}_max_rel_rise_2ndhalf"] = max_rel_h

        key = "val/gene_interaction/Pearson"
        if key in h:
            v = h[key].to_numpy(dtype=float)
            e = h["epoch"].to_numpy(dtype=float)
            m = np.isfinite(v)
            v, e = v[m], e[m]
            roll = pd.Series(v).rolling(ROLL_WINDOW, center=True, min_periods=1).mean()
            i = int(roll.idxmax())
            row["pearson_rollmax"] = float(roll.iloc[i])
            row["pearson_rollmax_epoch"] = float(e[i])
            row["pearson_roll_final"] = float(roll.iloc[-1])
            row["pearson_roll_drop_peak_to_final"] = float(roll.iloc[i] - roll.iloc[-1])
            row["pearson_raw_max"] = float(np.max(v))
            row["pearson_raw_max_epoch"] = float(e[int(np.argmax(v))])
            row["pearson_final"] = float(v[-1])
            if "val_loss_min_epoch" in row:
                row["gap_lossmin_minus_peak"] = (
                    float(row["val_loss_min_epoch"]) - float(e[i])
                )
        rows.append(row)

    df = pd.DataFrame(rows)
    out = osp.join(res, "check010_curve_detail.csv")
    df.to_csv(out, index=False)
    print(f"wrote {out}")
    with pd.option_context("display.width", 260, "display.max_columns", 60):
        print(
            df[
                [
                    c
                    for c in [
                        "run_id",
                        "last_epoch",
                        "val_loss_n_up",
                        "val_loss_n_steps",
                        "val_loss_max_rel_rise",
                        "val_loss_n_up_2ndhalf",
                        "val_loss_max_rel_rise_2ndhalf",
                        "train_loss_n_up",
                        "train_loss_max_rel_rise",
                        "train_loss_n_up_2ndhalf",
                        "val_loss_min_epoch",
                        "pearson_rollmax",
                        "pearson_rollmax_epoch",
                        "pearson_roll_final",
                        "pearson_roll_drop_peak_to_final",
                        "gap_lossmin_minus_peak",
                    ]
                    if c in df
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
