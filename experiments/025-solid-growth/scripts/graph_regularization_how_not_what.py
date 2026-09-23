#!/usr/bin/env python
# experiments/025-solid-growth/scripts/graph_regularization_how_not_what.py
# [[experiments.025-solid-growth.scripts.graph_regularization_how_not_what]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/025-solid-growth/scripts/graph_regularization_how_not_what
"""Draft figure: the graph penalty shapes HOW the transformer learns, not WHAT it predicts.

THE CLAIM THE FIGURE TESTS. Nine gene-gene graphs enter the Cell Graph Transformer only
as a KL prior on one attention layer. Five measurements, on two campaigns, say the graphs
never reach the prediction and yet decide whether training happens at all:

  a  In 010 the penalty was 99.99 percent of the training loss (an unintended x367 in the
     coefficient), and in 025, at the intended coefficient, it is 98 percent at epoch 0
     and settles near 31 percent. Removing it from a TRAINED checkpoint changes the
     predictions by at most 1.3e-6 against a spread of 0.036: adjacency is not an input
     to the prediction function.
  b  Removing it from TRAINING stops learning. Three 30-epoch runs at lambda = 1 reach
     0.456 to 0.464; three companions at lambda = 0 never leave 0.10.
  c  The regularized heads did learn the edges: neighbor recall at each gene's own degree
     is 0.73 to 1.00 on every graph.
  d  What the model predicts is close to additive. Its test predictions agree with the
     per-gene ridge at r = 0.73 to 0.76, and two transformer runs agree with each other at
     0.79 to 0.82, barely more.
  e  The same edges as a HARD mask, with no penalty term, change WHEN training happens:
     the mask placed exactly where 010 put its KL reaches 0.31 at epoch 1 and sits at
     the label mean for the remaining 50 epochs; the mask on layers 2 to 5 sits at the
     label mean for 20 epochs, lifts off at epoch 21, and reaches 0.42 by epoch 40 (job
     1607, 47 epochs, 12 h wall clock). The soft KL on the same build and split is at
     0.39 after epoch 1 and reaches 0.446. (An earlier draft of this figure, drawn when
     1607 had 10 epochs, called the layers-2-to-5 mask a failure; it is a 20-epoch delay.)
  f  The regimes on one axis. lambda = 0 fails, finite lambda trains from epoch 1, and the
     lambda -> infinity limit that Supplementary Note [note:graph-attention] proves is
     contained by the soft prior either fails (layer 1) or trains late and lands 0.025
     below the soft prior (layers 2 to 5). Held-out accuracy peaks at intermediate
     lambda, which is the Note's second predicted signature, seen on three points rather
     than a sweep.

WHAT IT DOES NOT SEPARATE, stated so the caption can. A penalty toward the biological
graphs conditions the optimization; a penalty toward degree-matched random graphs might
too. That control has not been run, and panel f leaves it as the open question.

PROVENANCE. Training curves and loss shares are pulled once from W&B by pinned run id and
frozen under results/graph_regularization_how_not_what/, so `--from-csv` re-renders with
no network. The 010 numbers in a and d are read from the committed result files of the
additive-baselines analysis in experiments/010-kuzmin-tmi/results/.

Run from repo root:
  python experiments/025-solid-growth/scripts/graph_regularization_how_not_what.py            # pull, freeze, render
  python experiments/025-solid-growth/scripts/graph_regularization_how_not_what.py --from-csv  # render from frozen CSVs
"""

import argparse
import os
import os.path as osp

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from matplotlib.ticker import MultipleLocator
from pydantic import BaseModel

from torchcell.timestamp import timestamp
from torchcell.utils import PANEL_WIDTHS_MM, PLOT_PALETTE, mm_to_in, savefig_true_size_svg

load_dotenv()
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
ASSET_IMAGES_DIR = os.environ["ASSET_IMAGES_DIR"]

RESULTS = osp.join(EXPERIMENT_ROOT, "025-solid-growth", "results", "graph_regularization_how_not_what")
RESULTS_010 = osp.join(EXPERIMENT_ROOT, "010-kuzmin-tmi", "results")
IMAGES = osp.join(ASSET_IMAGES_DIR, "025-solid-growth")

ENTITY = "zhao-group"
P010 = "torchcell_010-kuzmin-tmi_equivariant_cell_graph_transformer"
P025 = "torchcell_025-solid-growth_equivariant_cell_graph_transformer"


class RunSpec(BaseModel):
    """One pinned W&B run: what arm it is and how the figure should draw it."""

    run_id: str
    project: str
    arm: str  # soft_kl | no_penalty | hard_mask_L1 | hard_mask_L2_5 | ckpt
    campaign: str  # 010 | 025
    label: str
    job: str
    partial: bool = False  # still running or cut short; drawn dashed and said in the caption


RUNS: list[RunSpec] = [
    # 010, the matched 30-epoch pair: identical but for the graph term's weight.
    RunSpec(run_id="z9l5lesa", project=P010, arm="soft_kl", campaign="010", label="$\\lambda = 1$", job="2059261"),
    RunSpec(run_id="hl37p5kq", project=P010, arm="soft_kl", campaign="010", label="$\\lambda = 1$", job="2059266"),
    RunSpec(run_id="timffh8i", project=P010, arm="soft_kl", campaign="010", label="$\\lambda = 1$", job="2043575"),
    RunSpec(run_id="outxo94i", project=P010, arm="no_penalty", campaign="010", label="$\\lambda = 0$", job="2059260"),
    RunSpec(run_id="vjfp4d83", project=P010, arm="no_penalty", campaign="010", label="$\\lambda = 0$", job="2059267"),
    RunSpec(run_id="4js6ximz", project=P010, arm="no_penalty", campaign="010", label="$\\lambda = 0$", job="2059265"),
    # 010, the three reported checkpoints, for the loss share over a full run.
    RunSpec(run_id="lzs9pcj3", project=P010, arm="ckpt", campaign="010", label="M01", job="2027905"),
    RunSpec(run_id="yv4r30bi", project=P010, arm="ckpt", campaign="010", label="M02", job="2027907"),
    RunSpec(run_id="c7671wgj", project=P010, arm="ckpt", campaign="010", label="M03", job="2036902"),
    # 025, same build and split, the graph mechanism swapped.
    RunSpec(run_id="0yw7moue", project=P025, arm="soft_kl", campaign="025", label="soft KL, layer 1", job="1598"),
    RunSpec(run_id="7f1yrsq9", project=P025, arm="hard_mask_L1", campaign="025", label="hard mask, layer 1", job="1606"),
    RunSpec(run_id="ydxc0ts1", project=P025, arm="hard_mask_L2_5", campaign="025", label="hard mask, layers 2 to 5", job="1602", partial=True),
    RunSpec(run_id="4qmgkcgn", project=P025, arm="hard_mask_L2_5", campaign="025", label="hard mask, layers 2 to 5", job="1607"),
]
# The three re-evaluation runs that scored edge recovery on the 010 checkpoints.
EVAL_RUNS = {"M01": "leodrxht", "M02": "cvu2ryfw", "M03": "0psour3n"}

HIST_KEYS = [
    "epoch", "val/gene_interaction/Pearson", "train/point_loss", "train/graph_reg_loss",
    "train/total_loss", "train/norm_weighted_graph_reg", "train/transformed/gene_interaction/MSE",
]
# The six STRING v12.0 channels are labeled by channel; the axis note says they are STRING.
GRAPH_ORDER = [
    ("physical", "physical"), ("regulatory", "regulatory"), ("tflink", "TFLink"),
    ("string12_0_neighborhood", "neighborhood"), ("string12_0_fusion", "fusion"),
    ("string12_0_cooccurence", "cooccurrence"), ("string12_0_coexpression", "coexpression"),
    ("string12_0_experimental", "experimental"), ("string12_0_database", "database"),
]
ARM_COLOR = {
    "soft_kl": PLOT_PALETTE[0], "hard_mask_L1": PLOT_PALETTE[1],
    "hard_mask_L2_5": PLOT_PALETTE[2], "no_penalty": PLOT_PALETTE[5],
}
CKPT_COLOR = {"M01": PLOT_PALETTE[0], "M02": PLOT_PALETTE[1], "M03": PLOT_PALETTE[2]}


# --------------------------------------------------------------------------- data


def pull(api) -> None:
    """Freeze every pinned run's epoch-level history and the eval-run summaries."""
    os.makedirs(RESULTS, exist_ok=True)
    for spec in RUNS:
        r = api.run(f"{ENTITY}/{spec.project}/{spec.run_id}")
        h = r.history(samples=200_000, pandas=True)
        cols = [k for k in HIST_KEYS if k in h.columns]
        h = h[cols].apply(pd.to_numeric, errors="coerce")
        epoch_level(h).to_csv(osp.join(RESULTS, f"history_{spec.run_id}.csv"), index=False)
        print(f"froze {spec.campaign} {spec.arm:15s} {spec.run_id} ({len(h)} rows)")
    rows = []
    for tag, rid in EVAL_RUNS.items():
        s = dict(api.run(f"{ENTITY}/{P010}/{rid}").summary)
        for k, v in s.items():
            if ("edge_recovery" in k and k.endswith("recall_at_deg")) or "norm_weighted" in k \
                    or k.endswith("graph_reg_loss") or k.endswith("point_loss"):
                rows.append({"ckpt": tag, "run_id": rid, "key": k, "value": float(v)})
    pd.DataFrame(rows).to_csv(osp.join(RESULTS, "eval_summaries_010.csv"), index=False)
    pd.DataFrame([s.model_dump() for s in RUNS]).to_csv(osp.join(RESULTS, "runs.csv"), index=False)


def epoch_level(h: pd.DataFrame) -> pd.DataFrame:
    """One row per epoch: the validation metric as logged, the training terms averaged."""
    h = h.dropna(subset=["epoch"])
    g = h.groupby("epoch")
    out = pd.DataFrame({"epoch": sorted(h["epoch"].unique())}).set_index("epoch")
    if "val/gene_interaction/Pearson" in h:
        out["val_pearson"] = g["val/gene_interaction/Pearson"].last()
    for src, dst in [("train/point_loss", "point_loss"), ("train/graph_reg_loss", "graph_reg_loss"),
                     ("train/total_loss", "total_loss"), ("train/norm_weighted_graph_reg", "graph_share_logged"),
                     ("train/transformed/gene_interaction/MSE", "train_transformed_mse")]:
        if src in h:
            out[dst] = g[src].mean()
    if {"graph_reg_loss", "total_loss"} <= set(out.columns):
        out["graph_share"] = out["graph_reg_loss"] / out["total_loss"]
    return out.reset_index()


def load_histories() -> dict[str, pd.DataFrame]:
    return {s.run_id: pd.read_csv(osp.join(RESULTS, f"history_{s.run_id}.csv")) for s in RUNS}


def load_010_tables() -> dict[str, pd.DataFrame]:
    direct = pd.read_csv(osp.join(RESULTS_010, "cgt_direct_scoring.csv"))
    agree = pd.read_csv(osp.join(RESULTS_010, "paired_prediction_agreement.csv"))
    ladder = pd.read_csv(osp.join(RESULTS_010, "additive_baseline_gene_interaction.csv"))
    evals = pd.read_csv(osp.join(RESULTS, "eval_summaries_010.csv"))
    return {"direct": direct, "agree": agree, "ladder": ladder, "evals": evals}


# --------------------------------------------------------------------------- figure


def set_plot_style():
    plt.rcParams.update(
        {
            "font.family": "Arial", "font.size": 6, "axes.labelsize": 6, "axes.titlesize": 6,
            "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 5,
            "legend.title_fontsize": 5, "svg.fonttype": "none", "axes.linewidth": 0.5,
            "savefig.bbox": None, "savefig.pad_inches": 0.0,
        }
    )


def _letter(ax, letter):
    ax.text(-0.22, 1.05, letter, transform=ax.transAxes, fontsize=8, fontweight="bold",
            va="bottom", ha="left")


def _box(ax):
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.5)


def _first_epoch_above(h: pd.DataFrame, thr: float) -> int | None:
    """Epoch at which validation Pearson first exceeds `thr`; None if it never does."""
    above = h.dropna(subset=["val_pearson"]).query("val_pearson > @thr")
    return int(above.epoch.iloc[0]) if len(above) else None


def _pearson_axis(ax, lo=-0.1, hi=0.5):
    ax.set_ylim(lo, hi)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.tick_params(which="minor", length=0)
    ax.grid(True, which="both", axis="y", lw=0.3, color="0.85", zorder=0)
    ax.axhline(0, lw=0.4, color="black", zorder=1)


def plot(hist: dict[str, pd.DataFrame], t: dict[str, pd.DataFrame], out_stem: str) -> dict:
    set_plot_style()
    W = mm_to_in(PANEL_WIDTHS_MM["full"])
    fig, axes = plt.subplots(2, 3, figsize=(W, mm_to_in(95)))
    fig.subplots_adjust(left=0.06, right=0.995, top=0.93, bottom=0.11, wspace=0.42, hspace=0.75)
    (ax_a, ax_b, ax_c), (ax_d, ax_e, ax_f) = axes
    by_id = {s.run_id: s for s in RUNS}
    numbers: dict = {}

    # a: share of the training loss carried by the graph penalty, over epochs.
    for spec in RUNS:
        if spec.arm == "ckpt":
            h = hist[spec.run_id]
            ax_a.plot(h.epoch, h.graph_share_logged, lw=0.8, color=CKPT_COLOR[spec.label],
                      label=f"010 {spec.label}")
    h = hist["0yw7moue"]
    ax_a.plot(h.epoch, h.graph_share, lw=0.8, color=ARM_COLOR["soft_kl"], ls="--", label="025 soft KL")
    numbers["share_010_final"] = {s.label: float(hist[s.run_id].graph_share_logged.dropna().iloc[-1])
                                  for s in RUNS if s.arm == "ckpt"}
    numbers["share_025"] = {"epoch0": float(h.graph_share.iloc[0]), "epoch1": float(h.graph_share.iloc[1]),
                            "final": float(h.graph_share.dropna().iloc[-1])}
    swap = t["direct"].dropna(subset=["rmse"]).query("pearson.isna()", engine="python")["rmse"]
    numbers["inference_swap_max_abs_delta"] = swap.tolist()
    ax_a.set_ylim(0, 1.02)
    ax_a.yaxis.set_major_locator(MultipleLocator(0.2)); ax_a.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax_a.tick_params(which="minor", length=0)
    ax_a.grid(True, which="both", axis="y", lw=0.3, color="0.85", zorder=0)
    ax_a.set_xlabel("Epoch"); ax_a.set_ylabel("Graph penalty share of training loss")
    ax_a.legend(frameon=False, loc="center right")
    ax_a.text(0.98, 0.04, f"Removed at inference:\nmax |{chr(916)}prediction| {chr(8804)} {max(swap):.1e}\n(spread 0.036)",
              transform=ax_a.transAxes, ha="right", va="bottom", fontsize=5)
    _letter(ax_a, "a")

    # b: 010, the matched pair with the penalty on and off.
    for spec in RUNS:
        if spec.campaign == "010" and spec.arm in ("soft_kl", "no_penalty"):
            h = hist[spec.run_id].dropna(subset=["val_pearson"])
            ax_b.plot(h.epoch, h.val_pearson, lw=0.8, color=ARM_COLOR[spec.arm],
                      label=spec.label if spec.job in ("2059261", "2059260") else None)
    numbers["010_best_val"] = {s.run_id: float(hist[s.run_id].val_pearson.max())
                               for s in RUNS if s.campaign == "010" and s.arm != "ckpt"}
    _pearson_axis(ax_b)
    ax_b.set_xlabel("Epoch"); ax_b.set_ylabel("Validation Pearson")
    ax_b.legend(frameon=False, loc="center right", title="010, 30 epochs, n = 3 each")
    _letter(ax_b, "b")

    # c: the regularized heads recover their graphs.
    ev = t["evals"]
    ev = ev[ev.key.str.contains("recall_at_deg")].copy()
    ev["graph"] = ev.key.str.extract(r"val_edge_recovery/(.+)_L1_H\d/recall_at_deg")
    names = [g for g, _ in GRAPH_ORDER]
    for j, (g, disp) in enumerate(GRAPH_ORDER):
        sub = ev[ev.graph == g]
        for tag, col in CKPT_COLOR.items():
            v = sub[sub.ckpt == tag].value
            ax_c.scatter([j] * len(v), v, s=8, color=col, zorder=3, label=tag if j == 0 else None)
    numbers["edge_recovery_range"] = [float(ev.value.min()), float(ev.value.max())]
    ax_c.set_xticks(range(len(names)))
    ax_c.set_xticklabels([d for _, d in GRAPH_ORDER], rotation=45, ha="right", rotation_mode="anchor")
    ax_c.annotate("", xy=(2.6, 0.615), xytext=(8.4, 0.615),
                  arrowprops=dict(arrowstyle="-", lw=0.5, color="black"))
    ax_c.text(5.5, 0.625, "STRING v12.0 channels", ha="center", va="bottom", fontsize=5)
    ax_c.set_ylim(0.6, 1.02)
    ax_c.yaxis.set_major_locator(MultipleLocator(0.2)); ax_c.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax_c.tick_params(which="minor", length=0)
    ax_c.grid(True, which="both", axis="y", lw=0.3, color="0.85", zorder=0)
    ax_c.set_ylabel("Neighbor recall at own degree")
    ax_c.legend(frameon=False, loc="lower left", title="010 checkpoint, layer 1")
    _letter(ax_c, "c")

    # d: what the prediction is. Agreement between models' test predictions.
    ag = t["agree"]
    ag = ag[ag.quantity == "pred_pearson"]
    groups = [
        ("Additive ridge\nvs CGT", ag[ag.pair.str.startswith("B1_additive|CGT")].value.to_numpy()),
        ("Embedding MLP\nvs CGT", ag[ag.pair.str.startswith("B5_mlp|CGT")].value.to_numpy()),
        ("CGT\nvs CGT", ag[ag.pair.str.match(r"CGT_M0\d\|CGT_M0\d")].value.to_numpy()),
    ]
    for j, (lab, v) in enumerate(groups):
        ax_d.bar(j, v.mean(), width=0.6, color=PLOT_PALETTE[j], edgecolor="black", lw=0.5, zorder=2)
        ax_d.scatter(np.full(len(v), j) + np.linspace(-0.12, 0.12, len(v)), v, s=8,
                     facecolor="white", edgecolor="black", lw=0.5, zorder=3)
    numbers["agreement"] = {lab.replace("\n", " "): v.round(3).tolist() for lab, v in groups}
    ladder = t["ladder"]
    b1 = float(ladder.query("model == 'B1_additive_gene' and split == 'test'").pearson.iloc[0])
    cgt = t["direct"].query("split == 'test'").dropna(subset=["pearson"]).pearson
    numbers["ladder_test"] = {"B1": b1, "CGT": cgt.round(4).tolist()}
    ax_d.set_xticks(range(3)); ax_d.set_xticklabels([g for g, _ in groups])
    ax_d.set_ylim(0, 1.02)
    ax_d.yaxis.set_major_locator(MultipleLocator(0.2)); ax_d.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax_d.tick_params(which="minor", length=0)
    ax_d.grid(True, which="both", axis="y", lw=0.3, color="0.85", zorder=0)
    ax_d.set_ylabel("Pearson between test predictions")
    ax_d.text(0.03, 0.96, f"Test Pearson vs labels:\nadditive ridge {b1:.3f}\nCGT {cgt.min():.3f} to {cgt.max():.3f}",
              transform=ax_d.transAxes, ha="left", va="top", fontsize=5)
    _letter(ax_d, "d")

    # e: 025, the same edges as a soft prior and as a hard mask.
    seen = set()
    # Complete runs first so each arm's legend entry is drawn from a solid line.
    for spec in sorted((s for s in RUNS if s.campaign == "025"), key=lambda s: s.partial):
        h = hist[spec.run_id].dropna(subset=["val_pearson"])
        ax_e.plot(h.epoch, h.val_pearson, lw=0.8, color=ARM_COLOR[spec.arm],
                  ls="--" if spec.partial else "-",
                  label=None if spec.arm in seen else spec.label)
        seen.add(spec.arm)
    numbers["025"] = {s.run_id: {"best": float(hist[s.run_id].val_pearson.max()),
                                 "best_epoch": int(hist[s.run_id].loc[hist[s.run_id].val_pearson.idxmax(), "epoch"]),
                                 "last": float(hist[s.run_id].val_pearson.dropna().iloc[-1]),
                                 "epochs": int(hist[s.run_id].epoch.max()),
                                 "first_epoch_above_0.2": _first_epoch_above(hist[s.run_id], 0.2)}
                      for s in RUNS if s.campaign == "025"}
    _pearson_axis(ax_e)
    ax_e.set_xlabel("Epoch"); ax_e.set_ylabel("Validation Pearson")
    ax_e.legend(frameon=False, loc="center right", title="025, same build and split")
    _letter(ax_e, "e")

    # f: the three regimes on one axis, best epoch filled and last epoch hollow.
    cats = [("no_penalty", "none\n$\\lambda = 0$"), ("soft_kl", "soft KL\nfinite $\\lambda$"),
            ("hard_mask_L1", "hard mask\nlayer 1"), ("hard_mask_L2_5", "hard mask\nlayers 2 to 5")]
    for j, (arm, lab) in enumerate(cats):
        specs = [s for s in RUNS if s.arm == arm]
        for k, s in enumerate(specs):
            h = hist[s.run_id].dropna(subset=["val_pearson"])
            x = j + (k - (len(specs) - 1) / 2) * 0.13
            ax_f.scatter(x, h.val_pearson.max(), s=12, color=ARM_COLOR[arm], edgecolor="black", lw=0.4, zorder=3)
            ax_f.scatter(x, h.val_pearson.iloc[-1], s=12, facecolor="white", edgecolor=ARM_COLOR[arm], lw=0.7, zorder=3)
            ax_f.plot([x, x], [h.val_pearson.iloc[-1], h.val_pearson.max()], lw=0.5, color=ARM_COLOR[arm], zorder=2)
    ax_f.scatter([], [], s=12, color="0.4", edgecolor="black", lw=0.4, label="best epoch")
    ax_f.scatter([], [], s=12, facecolor="white", edgecolor="0.4", lw=0.7, label="last epoch")
    ax_f.set_xticks(range(len(cats))); ax_f.set_xticklabels([l for _, l in cats])
    ax_f.set_xlim(-0.6, len(cats) - 0.4)
    _pearson_axis(ax_f, -0.1, 0.64)
    ax_f.set_ylabel("Validation Pearson")
    ax_f.set_xlabel("How the edges enter training")
    ax_f.legend(frameon=False, loc="center right")
    ax_f.text(0.03, 0.97, "hard mask = $\\lambda \\to \\infty$ limit of the soft prior\n"
              "layers 2 to 5: flat for 20 epochs, then trains\n"
              "not run: KL toward degree-matched random graphs",
              transform=ax_f.transAxes, ha="left", va="top", fontsize=5, style="italic")
    _letter(ax_f, "f")

    for ax in axes.ravel():
        _box(ax)
    fig.savefig(out_stem + ".png", dpi=300)
    savefig_true_size_svg(fig, out_stem + ".svg")
    plt.close(fig)
    return numbers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-csv", action="store_true", help="render from the frozen CSVs, no W&B access")
    args = ap.parse_args()
    os.makedirs(IMAGES, exist_ok=True)
    if not args.from_csv:
        import wandb

        pull(wandb.Api(timeout=60))
    hist = load_histories()
    tables = load_010_tables()
    stem = osp.join(IMAGES, f"graph_regularization_how_not_what_{timestamp()}")
    numbers = plot(hist, tables, stem)
    import json

    with open(osp.join(RESULTS, "figure_numbers.json"), "w") as f:
        json.dump(numbers, f, indent=2)
    print(json.dumps(numbers, indent=2))
    print(f"\nwrote {stem}.{{png,svg}} and {RESULTS}/figure_numbers.json")


if __name__ == "__main__":
    main()
