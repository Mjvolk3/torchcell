# experiments/040-inhibitor-synergy-wetlab/scripts/r3_convergence.py
# [[experiments.040-inhibitor-synergy-wetlab.scripts.r3_convergence]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/040-inhibitor-synergy-wetlab/scripts/r3_convergence
"""Is the mixture trainer converged, per source, on every round-3 and round-4 run?

Reads every ``*_history.csv`` under ``results/mixture/`` (GilaHyper) and
``results/mixture-delta/`` (Delta) and answers three questions per run and per source:

* **training loss**: is it still falling at the end? The logged value is one step's
  loss at each evaluation, so it is smoothed with a three-point median before the
  slope over the last fifth of the epochs is read, relative to the level there.
* **held-out loss**: where does it bottom, what fraction of its first-evaluation value
  it has removed there (scale-free, since the held-out compounds' standardized
  variance differs by fold), and how far has it risen by the end?
* **held-out score**: where does the centered Spearman peak, and what does the
  validation selector lose against that peak?

Writes ``results/r3_convergence_runs.csv`` (one row per run and source) and
``results/r3_convergence_arms.csv`` (means over runs per arm), and draws the mean
curves of the headline arms, both machines pooled, to
``$ASSET_IMAGES_DIR/040-inhibitor-synergy-wetlab/r3_convergence_<timestamp>``.

    python experiments/040-inhibitor-synergy-wetlab/scripts/r3_convergence.py
"""

from __future__ import annotations

import glob
import os
import os.path as osp
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from torchcell.timestamp import timestamp  # noqa: E402

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from mixture_data import read_history  # noqa: E402

from torchcell.utils import (  # noqa: E402
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    apply_paper_style,
    mm_to_in,
    savefig_true_size_svg,
)

load_dotenv()
apply_paper_style()
EXPERIMENT = osp.dirname(osp.dirname(osp.abspath(__file__)))
RESULTS = osp.join(EXPERIMENT, "results")
SLUG = "040-inhibitor-synergy-wetlab"
OUT = osp.join(os.environ["ASSET_IMAGES_DIR"], SLUG)
MACHINES = {"gilahyper": "mixture", "delta": "mixture-delta"}
SOURCES = ["vanacloig", "hoepfner_hop", "hillenmeyer_het", "wildenhain"]
SOURCE_COLOR = dict(
    zip(SOURCES, [PLOT_PALETTE[0], PLOT_PALETTE[1], PLOT_PALETTE[2], PLOT_PALETTE[3]])
)
#: the arms whose mean curves are drawn, in the order of the panels' legends
DRAWN = [
    ("mix_van_e150", "one source"),
    ("mix_vanhop_e150", "two sources"),
    ("mix_vanhophet_e150", "three sources"),
    ("mix_vanhophet_e150_cb96", "three, 96 conditions per step"),
    ("mix_vanhophet_e150_aw0p3", "three, auxiliary weight 0.3"),
]
ARM_STYLE = dict(zip([a for a, _ in DRAWN], ["-", "--", ":", "-.", (0, (1, 1))]))


def arm_of(name: str) -> str:
    return re.sub(r"_s\d+$", "", re.sub(r"_f\d+$", "", name))


def smooth(x: np.ndarray) -> np.ndarray:
    """Three-point running median, the ends kept."""
    out = x.copy()
    for i in range(1, len(x) - 1):
        out[i] = np.median(x[i - 1 : i + 2])
    return out


def rel_slope(epoch: np.ndarray, y: np.ndarray) -> float:
    """Slope of ``y`` over the last fifth of the epochs, as a fraction of its level there."""
    cut = epoch >= epoch.max() - 0.2 * (epoch.max() - epoch.min())
    e, v = epoch[cut], y[cut]
    if len(e) < 3 or np.nanmean(v) == 0:
        return float("nan")
    b = np.polyfit(e, v, 1)[0]
    return float(b * (e.max() - e.min()) / np.nanmean(v))


def histories() -> list[tuple[str, str, str, pd.DataFrame]]:
    out = []
    for machine, sub in MACHINES.items():
        for path in sorted(
            glob.glob(osp.join(RESULTS, sub, "r[34]*", "*_history.csv"))
        ):
            sweep = osp.basename(osp.dirname(path))
            if "smoke" in sweep:
                continue
            name = osp.basename(path).replace("_history.csv", "")
            out.append((machine, sweep, name, read_history(path)))
    return out


def run_rows() -> pd.DataFrame:
    rows = []
    for machine, sweep, name, history in histories():
        for seed, run in history.groupby("seed"):
            run = run.sort_values("epoch")
            epoch = run["epoch"].to_numpy(dtype=float)
            test = run["score/test_centered_spearman"].to_numpy(dtype=float)
            chosen = int(run["selected_step"].iloc[0])
            at = run[run["step"] == chosen]
            base = {
                "machine": machine,
                "sweep": sweep,
                "name": name,
                "arm": arm_of(name),
                "seed": int(str(seed)),
                "fold": int(run["fold"].iloc[0]),
                "epochs": int(round(epoch.max())),
                "lr_end": float(run["opt/lr"].iloc[-1]),
                "test_peak": float(np.nanmax(test)),
                "test_peak_epoch": int(round(epoch[int(np.nanargmax(test))])),
                "test_end": float(test[-1]),
                "test_selected": float(at["score/test_centered_spearman"].iloc[0]),
                "selected_epoch": int(
                    round(chosen / (run["step"].max() / epoch.max()))
                ),
            }
            for source in SOURCES:
                tl, hl = f"train_loss/{source}", f"test_loss/{source}"
                if hl not in run or run[hl].isna().all():
                    continue
                held = run[hl].to_numpy(dtype=float)
                i_min = int(np.nanargmin(held))
                row = dict(base)
                row["source"] = source
                row["heldout_min"] = float(held[i_min])
                row["heldout_min_epoch"] = int(round(epoch[i_min]))
                row["heldout_end"] = float(held[-1])
                row["heldout_rise_after_min"] = float(held[-1] - held[i_min])
                row["heldout_removed_from_first"] = float(1.0 - held[i_min] / held[0])
                if tl in run and not run[tl].isna().all():
                    train = smooth(run[tl].to_numpy(dtype=float))
                    row["train_early"] = float(np.nanmedian(train[1:4]))
                    row["train_end"] = float(np.nanmedian(train[-3:]))
                    row["train_rel_slope_last_fifth"] = rel_slope(epoch, train)
                rows.append(row)
    return pd.DataFrame(rows)


def arm_table(runs: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "epochs",
        "train_early",
        "train_end",
        "train_rel_slope_last_fifth",
        "heldout_min",
        "heldout_min_epoch",
        "heldout_end",
        "heldout_rise_after_min",
        "heldout_removed_from_first",
        "test_peak",
        "test_peak_epoch",
        "test_selected",
        "selected_epoch",
        "test_end",
    ]
    g = runs.groupby(["arm", "source"])
    out = g[cols].mean()
    out["runs"] = g.size()
    out["train_still_falling_frac"] = g["train_rel_slope_last_fifth"].apply(
        lambda s: float((s < -0.05).mean())
    )
    return out.reset_index()


def mean_curve(arm: str, column: str) -> pd.Series | None:
    frames = []
    for machine, sweep, name, history in histories():
        if arm_of(name) != arm or column not in history:
            continue
        for _, run in history.groupby("seed"):
            s = run.set_index(run["epoch"].round().astype(int))[column]
            if s.isna().all():
                continue
            frames.append(
                smooth(s.to_numpy(dtype=float))
                if column.startswith("train_loss/")
                else s.to_numpy(dtype=float)
            )
    if not frames:
        return None
    n = min(len(f) for f in frames)
    stack = np.vstack([f[:n] for f in frames])
    epochs = np.arange(1, n + 1) * (150 // n if n else 1)
    return pd.Series(stack.mean(axis=0), index=epochs)


def draw(runs: pd.DataFrame) -> str:
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(62)),
        layout="constrained",
    )
    for arm, _ in DRAWN:
        for source in SOURCES:
            c = mean_curve(arm, f"train_loss/{source}")
            if c is not None:
                axes[0].plot(
                    c.index,
                    c.to_numpy(),
                    linestyle=ARM_STYLE[arm],
                    color=SOURCE_COLOR[source],
                    linewidth=0.8,
                )
            h = mean_curve(arm, f"test_loss/{source}")
            if h is not None:
                axes[1].plot(
                    h.index,
                    h.to_numpy(),
                    linestyle=ARM_STYLE[arm],
                    color=SOURCE_COLOR[source],
                    linewidth=0.8,
                )
        t = mean_curve(arm, "score/test_centered_spearman")
        if t is not None:
            axes[2].plot(
                t.index,
                t.to_numpy(),
                linestyle=ARM_STYLE[arm],
                color=PLOT_PALETTE[10],
                linewidth=0.8,
            )
        v = mean_curve(arm, "score/val_centered_spearman")
        if v is not None:
            axes[2].plot(
                v.index,
                v.to_numpy(),
                linestyle=ARM_STYLE[arm],
                color=PLOT_PALETTE[5],
                linewidth=0.6,
            )
    axes[0].set_title(
        "training loss per source (one step per evaluation, smoothed), mean over runs"
    )
    axes[0].set_ylabel("standardized MSE")
    axes[1].set_title("held-out loss per source, mean over runs")
    axes[1].axhline(1.0, color="black", linewidth=0.4)
    axes[2].set_title("held-out (dark) and validation (gray) centered Spearman")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.set_xlim(0, 150)
    axes[1].set_ylim(0.3, 1.8)
    handles = [
        Line2D([], [], color=SOURCE_COLOR[s], linewidth=0.8, label=s)
        for s in SOURCES[:3]
    ]
    handles += [
        Line2D([], [], color="black", linestyle=ARM_STYLE[a], linewidth=0.8, label=lab)
        for a, lab in DRAWN
    ]
    fig.legend(
        handles=handles, loc="outside lower center", ncol=4, fontsize=4.8, frameon=False
    )
    os.makedirs(OUT, exist_ok=True)
    stem = osp.join(OUT, f"r3_convergence_{timestamp()}")
    fig.savefig(stem + ".png", dpi=300)
    savefig_true_size_svg(fig, stem + ".svg")
    return stem + ".png"


def main() -> None:
    runs = run_rows()
    runs.to_csv(osp.join(RESULTS, "r3_convergence_runs.csv"), index=False)
    arms = arm_table(runs)
    arms.to_csv(osp.join(RESULTS, "r3_convergence_arms.csv"), index=False)
    pd.set_option("display.width", 260)
    print(arms.round(3).to_string(index=False))
    print(draw(runs))
    print(f"{len(runs)} run-source rows, {runs.name.nunique()} runs")


if __name__ == "__main__":
    main()
