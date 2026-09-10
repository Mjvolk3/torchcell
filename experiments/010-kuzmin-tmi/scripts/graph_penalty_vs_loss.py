# experiments/010-kuzmin-tmi/scripts/graph_penalty_vs_loss.py
# [[experiments.010-kuzmin-tmi.scripts.graph_penalty_vs_loss]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/010-kuzmin-tmi/scripts/graph_penalty_vs_loss
"""How much of the 010 training objective was the graph penalty, epoch by epoch.

The additive-baselines report states that the Kullback-Leibler graph penalty was 99.99
percent of what the optimizer minimized (``norm_weighted_graph_reg`` 0.99986, a graph
term of about 5,740 against a point loss of 0.81). Those are single numbers read off run
summaries. This script pulls the full per-step history of the three 010 training runs
from W&B and draws the three loss components over training, so the claim can be read
as a curve rather than a quote, and it sets the 025 replication (GilaHyper job 1598,
same model and config on the 025 build) beside them.

The loss the trainer logs decomposes as
    total = 1.0 * point_loss + 0.1 * dist_loss + weighted_graph_reg
with ``dist_loss`` logged UNWEIGHTED (the 0.1 is applied in ``total``), and
``norm_weighted_graph_reg`` = weighted_graph_reg / total. That identity is checked on
the pulled rows before anything is plotted.

Runs (entity zhao-group):
    M01  lzs9pcj3   torchcell_010-kuzmin-tmi_equivariant_cell_graph_transformer
    M02  yv4r30bi   same project
    M03  c7671wgj   same project
    1598 0yw7moue   torchcell_025-solid-growth_equivariant_cell_graph_transformer
                    (rank 0 of the 4-GPU job; the other three ranks log no val rows)

Train rows are one per optimizer step (the trainer logs each step); val rows are one
per epoch. Both are cached to
``results/graph_penalty_vs_loss_history.csv`` and read from there unless ``--refresh``.

Outputs
    results/graph_penalty_vs_loss_history.csv   tidy: run, stage, epoch, _step, metrics
    results/graph_penalty_vs_loss_summary.json  per run: epochs, shares, medians, best
    $ASSET_IMAGES_DIR/010-kuzmin-tmi/graph_penalty_vs_loss.{svg,png}

Run from the repo root:
    python experiments/010-kuzmin-tmi/scripts/graph_penalty_vs_loss.py [--refresh]
"""

import argparse
import json
import os
import os.path as osp

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from matplotlib.ticker import MultipleLocator

from torchcell.utils import (
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    apply_paper_style,
    mm_to_in,
    panel_label,
    savefig_true_size_svg,
)

load_dotenv()

EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
ASSET_IMAGES_DIR = os.environ["ASSET_IMAGES_DIR"]
RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "010-kuzmin-tmi", "results")
HISTORY_CSV = osp.join(RESULTS_DIR, "graph_penalty_vs_loss_history.csv")
SUMMARY_JSON = osp.join(RESULTS_DIR, "graph_penalty_vs_loss_summary.json")

ENTITY = "zhao-group"
P010 = "torchcell_010-kuzmin-tmi_equivariant_cell_graph_transformer"
P025 = "torchcell_025-solid-growth_equivariant_cell_graph_transformer"
RUNS = {
    "M01": (P010, "lzs9pcj3"),
    "M02": (P010, "yv4r30bi"),
    "M03": (P010, "c7671wgj"),
    "1598": (P025, "0yw7moue"),
}
METRICS = [
    "point_loss",
    "dist_loss",
    "graph_reg_loss",
    "weighted_graph_reg",
    "norm_weighted_graph_reg",
    "total_loss",
    "loss",
]
PEARSON = "val/gene_interaction/Pearson"
DIST_LAMBDA = 0.1


def pull_history() -> pd.DataFrame:
    """Every train step and every val epoch of each run, as one tidy frame.

    ``scan_history(keys=...)`` returns only rows carrying EVERY requested key, and the
    trainer writes train and val metrics on different rows, so the two stages are
    pulled separately and stacked.
    """
    import wandb

    api = wandb.Api(timeout=300)
    frames = []
    for label, (project, run_id) in RUNS.items():
        run = api.run(f"{ENTITY}/{project}/{run_id}")
        for stage in ("train", "val"):
            keys = ["epoch", "_step"] + [f"{stage}/{m}" for m in METRICS]
            if stage == "val":
                keys.append(PEARSON)
            hist = pd.DataFrame(list(run.scan_history(keys=keys, page_size=5000)))
            if hist.empty:
                raise RuntimeError(f"{label} {run_id}: no {stage} rows for {keys}")
            hist = hist.rename(columns={f"{stage}/{m}": m for m in METRICS})
            if stage == "val":
                hist = hist.rename(columns={PEARSON: "pearson"})
            else:
                hist["pearson"] = np.nan
            hist.insert(0, "stage", stage)
            hist.insert(0, "run_id", run_id)
            hist.insert(0, "run", label)
            frames.append(hist)
            print(f"{label} {run_id} {stage}: {len(hist)} rows, state {run.state}")
    df = pd.concat(frames, ignore_index=True).sort_values(["run", "stage", "_step"])
    return df.reset_index(drop=True)


def check_decomposition(df: pd.DataFrame) -> None:
    """Total = point + 0.1 * dist + weighted graph on every row, and define the share.

    The share plotted and summarized is ``graph_share`` = weighted_graph_reg / total_loss
    recomputed from the logged terms, not the logged ``norm_weighted_graph_reg``. On the
    single-GPU 010 runs the two agree to 6e-7. On the 4-GPU 1598 run every logged value
    is a mean over ranks (``sync_dist``), so the logged share is a mean of per-rank
    ratios and differs from the ratio of the means by up to 0.08 on a single step. The
    recomputed quantity is the same statistic for every run.
    """
    recon = df["point_loss"] + DIST_LAMBDA * df["dist_loss"] + df["weighted_graph_reg"]
    rel = ((recon - df["total_loss"]).abs() / df["total_loss"]).max()
    print(
        f"decomposition check: max relative error of point + 0.1 dist + graph = {rel:.2e}"
    )
    assert rel < 1e-4, "total_loss is not point + 0.1 dist + graph on these rows"
    df["graph_share"] = df["weighted_graph_reg"] / df["total_loss"]
    gap = (df["graph_share"] - df["norm_weighted_graph_reg"]).abs()
    for run, g in gap.groupby(df["run"]):
        print(f"  {run}: logged share vs graph/total, max |diff| {g.max():.2e}")


def per_epoch(df: pd.DataFrame) -> pd.DataFrame:
    """Train rows averaged within each epoch; val rows are already one per epoch."""
    tr = (
        df[df.stage == "train"]
        .groupby(["run", "epoch"], as_index=False)[METRICS]
        .mean()
        .assign(stage="train")
    )
    # Share of the epoch's summed loss, not the mean of per-step shares.
    tr["graph_share"] = tr["weighted_graph_reg"] / tr["total_loss"]
    va = df[df.stage == "val"][
        ["run", "epoch", "stage", "pearson", "graph_share"] + METRICS
    ]
    return pd.concat([tr, va], ignore_index=True)


def summarize(ep: pd.DataFrame) -> dict:
    out = {}
    for run in RUNS:
        tr = ep[(ep.run == run) & (ep.stage == "train")].sort_values("epoch")
        va = ep[(ep.run == run) & (ep.stage == "val")].sort_values("epoch")
        best = va.loc[va["pearson"].idxmax()]
        out[run] = {
            "run_id": RUNS[run][1],
            "project": RUNS[run][0],
            "epochs_logged": int(va["epoch"].max()) + 1,
            "train_share_median": float(tr["graph_share"].median()),
            "train_share_final": float(tr["graph_share"].iloc[-1]),
            "train_share_min": float(tr["graph_share"].min()),
            "train_point_loss_median": float(tr["point_loss"].median()),
            "train_graph_term_median": float(tr["weighted_graph_reg"].median()),
            "train_graph_term_final": float(tr["weighted_graph_reg"].iloc[-1]),
            "val_best_pearson": float(best["pearson"]),
            "val_best_epoch": int(best["epoch"]),
            "val_point_loss_at_best": float(best["point_loss"]),
            "val_graph_term_at_best": float(best["weighted_graph_reg"]),
            "val_share_at_best": float(best["graph_share"]),
            "val_point_loss_final": float(va["point_loss"].iloc[-1]),
            "val_graph_term_final": float(va["weighted_graph_reg"].iloc[-1]),
            "val_share_final": float(va["graph_share"].iloc[-1]),
        }
    return out


def _box(ax: plt.Axes) -> None:
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.5)


def plot(ep: pd.DataFrame, summary: dict) -> None:
    apply_paper_style()
    fig, axes = plt.subplots(
        2, 2, figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(105.0))
    )
    comp = [
        ("point_loss", 1.0, "point loss (MSE, weight 1)", PLOT_PALETTE[0]),
        ("dist_loss", DIST_LAMBDA, "0.1 x Sinkhorn term", PLOT_PALETTE[1]),
        ("weighted_graph_reg", 1.0, "graph penalty (weighted)", PLOT_PALETTE[2]),
    ]
    run_color = {
        "M01": PLOT_PALETTE[0],
        "M02": PLOT_PALETTE[1],
        "M03": PLOT_PALETTE[2],
        "1598": PLOT_PALETTE[3],
    }
    run_name = {
        "M01": "M01 lzs9pcj3",
        "M02": "M02 yv4r30bi",
        "M03": "M03 c7671wgj",
        "1598": "025 replication, job 1598",
    }

    # ---- (a) 010 loss components per epoch, M03 solid, M01/M02 light -------------
    ax = axes[0, 0]
    for run, alpha_key in (("M01", 1), ("M02", 1), ("M03", 0)):
        tr = ep[(ep.run == run) & (ep.stage == "train")].sort_values("epoch")
        for key, w, name, color in comp:
            # M01 and M02 in the muted-earth sibling of the same slot (palette 13-18),
            # so the hue still names the term and M03 is the line that reads first.
            muted = PLOT_PALETTE[PLOT_PALETTE.index(color) + 12]
            ax.plot(
                tr["epoch"],
                w * tr[key],
                color=color if run == "M03" else muted,
                lw=1.0 if run == "M03" else 0.7,
                label=name if run == "M03" else None,
                zorder=3 if run == "M03" else 2,
            )
    ax.plot(
        [], [], color=PLOT_PALETTE[17], lw=0.7, label="same terms, M01 and M02 (muted)"
    )
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training loss term (epoch mean)")
    ax.set_title(
        "010 training runs M01 to M03: the three loss terms", fontsize=6, pad=3
    )
    ax.legend(loc="center right", handlelength=1.4, borderpad=0.3, labelspacing=0.25)
    _box(ax)

    # ---- (b) the same three terms for the 025 replication ---------------------------
    ax = axes[0, 1]
    tr = ep[(ep.run == "1598") & (ep.stage == "train")].sort_values("epoch")
    for key, w, name, color in comp:
        ax.plot(tr["epoch"], w * tr[key], color=color, lw=1.0, label=name)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training loss term (epoch mean)")
    ax.set_title(
        "025 replication, job 1598 (edge normalization fixed)", fontsize=6, pad=3
    )
    ax.legend(loc="upper right", handlelength=1.4, borderpad=0.3, labelspacing=0.25)
    _box(ax)

    # ---- (c) share of the total loss carried by the graph term ------------------------
    ax = axes[1, 0]
    for run in RUNS:
        tr = ep[(ep.run == run) & (ep.stage == "train")].sort_values("epoch")
        ax.plot(
            tr["epoch"],
            tr["graph_share"],
            color=run_color[run],
            lw=1.0,
            label=run_name[run],
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Graph term / total loss (train)")
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.tick_params(axis="y", which="minor", length=0)
    ax.grid(axis="y", which="both", lw=0.3, alpha=0.35)
    ax.set_title("share of the training objective in the graph term", fontsize=6, pad=3)
    # Aligned columns for the compared numbers, in a clear region (the 010 lines sit at
    # 1.0 and the 1598 line settles near 0.3, so 0.45 to 0.9 is empty after epoch 5).
    cols = (0.42, 0.66, 0.84)
    y0, dy = 0.80, 0.085
    for x, head in zip(cols, ("run", "median", "final")):
        ax.text(
            x,
            y0,
            head,
            transform=ax.transAxes,
            fontsize=6,
            ha="left",
            va="center",
            fontweight="bold",
            zorder=6,
        )
    for i, run in enumerate(RUNS):
        y = y0 - (i + 1) * dy
        ax.plot(
            [0.36, 0.40],
            [y, y],
            transform=ax.transAxes,
            color=run_color[run],
            lw=1.0,
            zorder=6,
        )
        s = summary[run]
        for x, txt in zip(
            cols,
            (run, f"{s['train_share_median']:.5f}", f"{s['train_share_final']:.5f}"),
        ):
            ax.text(
                x,
                y,
                txt,
                transform=ax.transAxes,
                fontsize=6,
                ha="left",
                va="center",
                zorder=6,
            )
    _box(ax)

    # ---- (d) validation Pearson per epoch, best epoch marked -----------------------
    ax = axes[1, 1]
    for run in RUNS:
        va = ep[(ep.run == run) & (ep.stage == "val")].sort_values("epoch")
        ax.plot(
            va["epoch"],
            va["pearson"],
            color=run_color[run],
            lw=1.0,
            label=run_name[run],
        )
        s = summary[run]
        ax.scatter(
            [s["val_best_epoch"]],
            [s["val_best_pearson"]],
            s=18,
            marker="D",
            facecolor=run_color[run],
            edgecolor="black",
            lw=0.5,
            zorder=5,
        )
    ax.scatter(
        [],
        [],
        s=18,
        marker="D",
        facecolor="white",
        edgecolor="black",
        lw=0.5,
        label="best-Pearson epoch",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Pearson r")
    ax.set_ylim(0, 0.5)
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.grid(axis="y", which="major", lw=0.3, alpha=0.35)
    ax.set_title("validation Pearson per epoch", fontsize=6, pad=3)
    ax.legend(loc="lower right", handlelength=1.4, borderpad=0.3, labelspacing=0.25)
    _box(ax)

    for ax, letter in zip(axes.ravel(), "abcd"):
        panel_label(ax, letter)
    fig.tight_layout(pad=0.4)

    stem = osp.join(ASSET_IMAGES_DIR, "010-kuzmin-tmi", "graph_penalty_vs_loss")
    os.makedirs(osp.dirname(stem), exist_ok=True)
    fig.savefig(stem + ".png", dpi=300)
    savefig_true_size_svg(fig, stem + ".svg")
    print(f"wrote {stem}.svg")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refresh", action="store_true", help="re-pull from W&B")
    args = ap.parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if args.refresh or not osp.exists(HISTORY_CSV):
        df = pull_history()
        df.to_csv(HISTORY_CSV, index=False)
        print(f"wrote {HISTORY_CSV}")
    else:
        df = pd.read_csv(HISTORY_CSV)
        print(f"read {HISTORY_CSV} ({len(df)} rows)")

    check_decomposition(df)
    ep = per_epoch(df)
    summary = summarize(ep)
    with open(SUMMARY_JSON, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"wrote {SUMMARY_JSON}\n")

    cols = [
        "epochs_logged",
        "train_share_median",
        "train_share_final",
        "train_point_loss_median",
        "train_graph_term_median",
        "val_best_pearson",
        "val_best_epoch",
        "val_point_loss_at_best",
        "val_graph_term_at_best",
        "val_share_at_best",
    ]
    print(pd.DataFrame(summary).T[cols].to_string(float_format=lambda v: f"{v:.5g}"))
    plot(ep, summary)


if __name__ == "__main__":
    main()
