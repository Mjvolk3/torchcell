# experiments/025-solid-growth/scripts/fig2_gene_interactions.py
# [[experiments.025-solid-growth.scripts.fig2_gene_interactions]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/025-solid-growth/scripts/fig2_gene_interactions
"""Figure 2, gene-gene interactions, as one full-width sheet in the figure-gate style.

Sixteen panels, four by four, no frames, every panel letter followed by the section
mark (a red cross: not yet reviewed by the author) and the figure mark (a full circle:
every number comes from a committed result file; a half circle: partial; an open circle:
idea only). Every number is read from a result file written by a committed script; the
file and script are named per panel below. Written to $ASSET_IMAGES_DIR/025-solid-growth/
fig2_gene_interactions.{svg,png} at PANEL_WIDTHS_MM["full"].

  a  the hand-drawn explainer, exported from the draw.io figure
     (notes/assets/drawio/Fig2-gene-interactions.explainer.py -> fig2_explainer.png)
  b  measured trigenic interaction by bin of the prediction, CGT checkpoint M03 on the
     37,673 test triples (010 results: cgt_predictions_M03_c7671wgj_test.npy,
     cgt_record_rows_test.npy; labels from the 010 build's label_df.parquet)
  c  the state of the art: Yeast9 FBA, DCell, DANGO, CGT, each as its replicates, and the
     no-interaction ridge as a line (SI notes si-note-fba, si-note-dcell-training,
     si-note-dango-full; 006 results dcell_training/checkpoints.csv)
  d  the additive null on the random split and on held-out query pairs (010 results:
     additive_baseline_gene_interaction.csv, query_pair_disjoint_cv.csv; the transformer
     on the disjoint split from 025 results: disjoint_embedding_readout.csv; the 030
     closure arm on that split from s3q_030_readout.csv, one seed, in flight)
  e  the nine graphs absent, as a hard mask, as a soft prior, and toward random graphs
     (025 results: graph_reg_sweep_runs.csv, graph_reg_sweep_summary.json)
  f  the training pool: S0 triples, S0 with the fitness head, the S3 closure
     (025 results: s3_closure_readout.csv)
  g  the prior's weight ladder, epoch 29, no penalty to lambda 100 and the mask
  h  biological minus random graphs, paired by seed, at each lambda the control ran
  i  validation Pearson by epoch: no penalty, mask, prior, random graphs
  j  edge recall at degree against each arm's own target, epoch 29
  k  sixty epochs: no penalty keeps declining, the prior holds
  l  the composite sequence embedding trains only with the prior
  m  the encoder's wild-type gene map colored by whether a gene appears in any training
     triple (025 results: wt_gene_embeddings/cluster_assignments.csv, go_enrichment.csv,
     summary.json, from wt_gene_embedding_clusters.py on the S3 closure checkpoint)
  n  the same map colored by Leiden cluster
  o  the best Gene Ontology term of every cluster
  p  the raw learnable gene table clustered the same way, as the control

Palette, primaries only: gray no graphs or a no-interaction null; red the cell graph
transformer with its soft prior (TorchCell); orange the hard mask; purple degree-matched
random graphs; blue a learned nonlinear null; yellow the other published models and the
pair or fitness-head variants. Arial 6 pt, true-size SVG, no bbox_inches="tight".
"""

from __future__ import annotations

import json
import os
import os.path as osp
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator
from scipy import stats

from torchcell.timestamp import timestamp
from torchcell.utils import PANEL_WIDTHS_MM, PLOT_PALETTE, PLOT_PALETTE_FILL, mm_to_in, savefig_true_size_svg

load_dotenv()
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
ASSET_IMAGES_DIR = os.environ["ASSET_IMAGES_DIR"]
DATA_ROOT = os.environ["DATA_ROOT"]
R010 = osp.join(EXPERIMENT_ROOT, "010-kuzmin-tmi", "results")
R006 = osp.join(EXPERIMENT_ROOT, "006-kuzmin-tmi", "results", "dcell_training")
R025 = osp.join(EXPERIMENT_ROOT, "025-solid-growth", "results")
BUILD_010 = osp.join(DATA_ROOT, "data/torchcell/experiments/010-kuzmin-tmi/001-small-build")
IMG_DIR = osp.join(ASSET_IMAGES_DIR, "025-solid-growth")

ORANGE, RED, PURPLE, YELLOW, BLUE, GRAY = PLOT_PALETTE[:6]
GRID = "#E3E3E3"
LIGHT = "#8A8A8A"
STATUS_SYMBOL = {"ready": "●", "partial": "◐", "todo": "○"}
STATUS_WORD = {"ready": "on file", "partial": "partial", "todo": "idea only"}
CROSS = ("✗", "#E53935")  # section status: not yet reviewed by the author
MAP = "tsne"  # which 2-D map the gene panels draw: "tsne" or "umap"
MAP_NAME = {"tsne": "t-SNE", "umap": "UMAP"}[MAP]
STATUS_MARKS = True  # the review marks beside each panel letter
STATUS_KEY = False  # their key at the foot of the sheet; off on the author's request (2026-10-13)

plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 6,
        "axes.labelsize": 6,
        "axes.titlesize": 6,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 5,
        "svg.fonttype": "none",
        "axes.linewidth": 0.5,
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic",
    }
)

# --------------------------------------------------------------------------- helpers


def strip(ax: Axes, grid: str = "y") -> None:
    """No frame: left and bottom hairlines only, a light grid behind the data."""
    for name, s in ax.spines.items():
        s.set_visible(name in ("left", "bottom"))
        s.set_linewidth(0.5)
    ax.tick_params(width=0.5, length=2)
    if grid:
        ax.grid(axis=grid, which="major", color=GRID, linewidth=0.3)
    ax.set_axisbelow(True)


def bars(ax: Axes, xs: list[float], groups: list[np.ndarray], colors: list[str], width: float = 0.62) -> None:
    """Bar is the mean, whisker the sample sd, every replicate an open circle; a single
    value has no whisker."""
    for x, vals, color in zip(xs, groups, colors):
        vals = np.asarray(vals, dtype=float)
        sd = vals.std(ddof=1) if len(vals) > 1 else 0.0
        ax.bar(
            x, vals.mean(), width=width, color=color, edgecolor="black", linewidth=0.4,
            yerr=sd if len(vals) > 1 else None,
            error_kw={"elinewidth": 0.5, "capsize": 1.2, "capthick": 0.5}, zorder=3,
        )
        jit = np.linspace(-0.25, 0.25, len(vals)) * width if len(vals) > 1 else np.array([0.0])
        ax.scatter(x + jit, vals, s=6, facecolor="white", edgecolor="black", linewidth=0.4, zorder=4)


def not_run(ax: Axes, x: float, y: float, text: str, color: str) -> None:
    ax.scatter([x], [y], s=12, marker="x", color=color, linewidth=0.6, zorder=4)
    ax.annotate(text, (x, y), xytext=(0, 7), textcoords="offset points", ha="center", va="bottom", fontsize=5, color="black")


def stars(p: float | None) -> str:
    if p is None or np.isnan(p):
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def curve(ax: Axes, hist: pd.DataFrame, arm: str, key: str, color: str, ls: str, label: str, z: int) -> None:
    sub = hist[(hist.arm == arm) & (hist.key == key)]
    g = sub.groupby("epoch")["value"]
    mean, sd, n = g.mean(), g.std(ddof=1), g.count()
    ax.plot(mean.index, mean.to_numpy(), color=color, lw=0.9, ls=ls, label=label, zorder=z)
    if (n.max() > 1) and sd.notna().any():
        ax.fill_between(mean.index, (mean - sd).to_numpy(), (mean + sd).to_numpy(), color=color, alpha=0.15, lw=0, zorder=z - 10)


def letters_status(fig: Figure, axes: list[Axes], statuses: list[str]) -> None:
    """Bold lowercase letter, the section cross, the figure mark, at each panel's top left."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    tops = {id(ax): ax.get_tightbbox(renderer).y1 for ax in axes}
    near = 0.06 * fig.bbox.height
    # One x per COLUMN: the leftmost extent of any panel in that column, so a panel with
    # no axis label (a, the maps) carries its letter where its neighbors below do.
    ncol = 4
    col_x0 = {}
    for k, ax in enumerate(axes):
        c = k % ncol
        col_x0[c] = min(col_x0.get(c, 1e9), ax.get_tightbbox(renderer).x0)
    for k, (ax, ch, status) in enumerate(zip(axes, "abcdefghijklmnop", statuses)):
        box = ax.get_tightbbox(renderer)
        same = [tops[id(o)] for o in axes if abs(tops[id(o)] - tops[id(ax)]) < near]
        x0, y1 = inv.transform((col_x0[k % ncol], max(same)))
        x0 = max(x0, 0.004)
        y = y1 + 0.004
        fig.text(x0, y, ch, fontsize=8, fontweight="bold", ha="left", va="bottom")
        if not STATUS_MARKS:
            continue
        fig.text(x0 + 0.013, y + 0.001, CROSS[0], fontsize=5, color=CROSS[1], ha="left", va="bottom", family="DejaVu Sans")
        fig.text(x0 + 0.022, y + 0.001, STATUS_SYMBOL[status], fontsize=5, color=LIGHT, ha="left", va="bottom", family="DejaVu Sans")


def keys(fig: Figure, y_key: float) -> None:
    """Color key between rows 1 and 2, beside the panels that use it; status key at the foot."""
    handles = [
        Patch(facecolor=GRAY, edgecolor="black", linewidth=0.4, label="no graphs, or a no-interaction null"),
        Patch(facecolor=RED, edgecolor="black", linewidth=0.4, label="cell graph transformer, soft prior"),
        Patch(facecolor=ORANGE, edgecolor="black", linewidth=0.4, label="hard mask"),
        Patch(facecolor=PURPLE, edgecolor="black", linewidth=0.4, label="degree-matched random graphs"),
        Patch(facecolor=BLUE, edgecolor="black", linewidth=0.4, label="learned null (MLP)"),
        Patch(facecolor=YELLOW, edgecolor="black", linewidth=0.4, label="other arm"),
    ]
    fig.legend(handles=handles, loc="center", bbox_to_anchor=(0.52, y_key), ncol=6, frameon=False, fontsize=5, handlelength=1.2, handletextpad=0.4, columnspacing=1.2, borderpad=0)
    if not STATUS_KEY:
        return
    x = 0.03
    fig.text(x, 0.004, "section status", fontsize=5, color=LIGHT, ha="left", va="bottom", style="italic")
    fig.text(x + 0.075, 0.004, CROSS[0], fontsize=5, color=CROSS[1], ha="left", va="bottom", family="DejaVu Sans")
    fig.text(x + 0.087, 0.004, "not reviewed, none approved", fontsize=5, color=LIGHT, ha="left", va="bottom")
    x = 0.33
    fig.text(x, 0.004, "figure status", fontsize=5, color=LIGHT, ha="left", va="bottom", style="italic")
    x += 0.07
    for key in ("ready", "partial", "todo"):
        fig.text(x, 0.004, STATUS_SYMBOL[key], fontsize=5, color=LIGHT, ha="left", va="bottom", family="DejaVu Sans")
        fig.text(x + 0.012, 0.004, STATUS_WORD[key], fontsize=5, color=LIGHT, ha="left", va="bottom")
        x += 0.065


# ---------------------------------------------------------------------------- panels


def panel_a(ax: Axes) -> None:
    """The hand-drawn explainer, exported from the draw.io figure by
    notes/assets/drawio/Fig2-gene-interactions.explainer.py."""
    img = plt.imread(osp.join(IMG_DIR, "fig2_explainer.png"))
    ax.imshow(img, interpolation="lanczos")
    ax.set_anchor("NW")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title("what the lower orders do not explain")


def panel_b(ax: Axes) -> None:
    label_df = pd.read_parquet(osp.join(BUILD_010, "processed", "label_df.parquet"))
    record_ids = np.sort(label_df["index"].to_numpy())
    y_all = label_df.set_index("index").loc[record_ids, "gene_interaction"].to_numpy()
    rows = np.load(osp.join(R010, "cgt_record_rows_test.npy"))
    y = y_all[rows]
    p = np.load(osp.join(R010, "cgt_predictions_M03_c7671wgj_test.npy"))
    edges = [-np.inf, -0.40, -0.32, -0.24, -0.16, -0.08, 0.0, 0.08, 0.16, 0.24, np.inf]
    idx = np.digitize(p, edges[1:-1])
    groups = [y[idx == k] for k in range(len(edges) - 1)]
    # The look of the DCell and DANGO figures: boxes abutting, filled in the red fill,
    # outline and whiskers one dark-red line with no caps, the median in red, no fliers.
    dark_red = PLOT_PALETTE[7]
    ax.boxplot(
        groups, positions=range(len(groups)), widths=0.96, showfliers=False, patch_artist=True,
        showcaps=False,
        medianprops={"color": RED, "lw": 1.0}, whiskerprops={"color": dark_red, "lw": 0.8, "ls": "-"},
        boxprops={"facecolor": PLOT_PALETTE_FILL[1], "edgecolor": dark_red, "lw": 0.8}, zorder=3,
    )
    ax.axhline(0, color="black", lw=0.5, zorder=2)
    ax.axvline(5.5, color="black", lw=0.5, zorder=2)
    ax.set_xticks(np.arange(len(edges)) - 0.5)
    ax.set_xticklabels(["\u2212Inf", "\u22120.40", "\u22120.32", "\u22120.24", "\u22120.16", "\u22120.08", "0.00", "0.08", "0.16", "0.24", "Inf"], fontsize=5, rotation=45, ha="right")
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.set_xlabel("predicted genetic interaction")
    ax.set_ylabel("measured genetic interaction")
    r = stats.pearsonr(p, y)[0]
    ax.set_title(f"monotone across bins, r = {r:.3f}")
    strip(ax, grid="both")


def panel_c(ax: Axes) -> None:
    dcell = np.loadtxt(osp.join(R006, "checkpoints.csv"), delimiter=",", skiprows=1, usecols=1)
    # SOURCE: paper/nature-biotech/sections/si-note-fba.tex (scripted value 0.0019, n 332,313;
    # the 0.0006 once printed does not reproduce), si-note-dango-full.tex (five v9.1 runs,
    # best validation r), si-note-dcell-training.tex (three checkpoints of one run, upward
    # biased), trigenic_tau_model_comparison.py (three CGT replicates, max over epochs).
    models = [
        ("Yeast9\nFBA", np.array([0.0019]), YELLOW),
        ("DCell", dcell, PURPLE),
        ("DANGO", np.array([0.36759, 0.36708, 0.36637, 0.3658, 0.3652]), ORANGE),
        ("CGT", np.array([0.462, 0.452, 0.447]), RED),
    ]
    bars(ax, list(range(4)), [m[1] for m in models], [m[2] for m in models])
    add = pd.read_csv(osp.join(R010, "additive_baseline_gene_interaction.csv"))
    ridge = float(add[(add.model == "B1_additive_gene") & (add.split == "test")].pearson.iloc[0])
    ax.axhline(ridge, color=GRAY, lw=0.7, ls="--", zorder=2)
    ax.text(-0.45, ridge - 0.008, f"ridge, {ridge:.2f}", ha="left", va="top", fontsize=5, color="black")
    for x, (_, vals, _) in enumerate(models):
        # An opaque box so the DANGO value reads over the ridge line it sits on.
        ax.text(x, vals.max() + 0.015, f"{vals.mean():.3f}" if vals.mean() > 0.01 else f"{vals.mean():.4f}", ha="center", va="bottom", fontsize=5, zorder=6, bbox=dict(facecolor="white", edgecolor="none", pad=0.6))
    ax.set_xticks(range(4))
    ax.set_xticklabels([m[0] for m in models])
    ax.set_ylim(0, 0.52)
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.set_ylabel("Pearson r, trigenic interaction")
    ax.set_title("learned models lead; the ridge is at 0.40")
    strip(ax)


def panel_d(ax: Axes) -> None:
    rnd = pd.read_csv(osp.join(R010, "additive_baseline_gene_interaction.csv"))
    cv = pd.read_csv(osp.join(R010, "query_pair_disjoint_cv.csv"))
    # The transformer on the query-pair-disjoint split: the learnable-table control of the
    # 025 disjoint study (cgt_s0_q_kl_ctrl_016, three seeds on one split, validation Pearson
    # at epoch 29, the sheet's fixed reading), from disjoint_embedding_readout.csv. The
    # baselines on that split are test Pearson over five folds, so the two are drawn on one
    # axis with the difference named in the panel.
    dq = pd.read_csv(osp.join(R025, "disjoint_embedding_readout.csv"))
    cgt_q = dq[dq.config == "cgt_s0_q_kl_ctrl_016"].val_pearson_epoch29.dropna().to_numpy(dtype=float)
    learners = [
        ("ridge, genes", "B1_additive_gene", GRAY),
        ("ridge, genes + pairs", "B2_additive_plus_pair", YELLOW),
        ("MLP, genes only", "B5_gene_embedding_mlp", BLUE),
        ("CGT", "CGT_", RED),
    ]
    w = 0.19
    for g, group in enumerate(("random", "disjoint")):
        for k, (label, prefix, color) in enumerate(learners):
            x = g + (k - 1.5) * w
            if group == "random":
                vals = rnd[(rnd.split == "test") & rnd.model.str.startswith(prefix)].pearson.to_numpy(dtype=float)
            elif prefix == "CGT_":
                vals = cgt_q
            else:
                vals = cv[cv.model.str.startswith(prefix)].pearson.to_numpy(dtype=float)
            bars(ax, [x], [vals], [color], width=w * 0.9)
            print(f"d {group:8s} {label:22s} n={len(vals)} mean {vals.mean():.3f}")
    # The 030 closure arm on the same question (S3Q pool, composite embedding, fitness
    # co-training, dataset token; one seed, in flight on IGB mmli): the latest epoch synced,
    # from s3q_030_readout.csv, hatched because it is a partial reading of one seed.
    q030 = pd.read_csv(osp.join(R025, "s3q_030_readout.csv")).sort_values("epoch")
    last = q030.iloc[-1]
    x030 = 1 + 2.6 * w
    ax.bar(x030, last.val_pearson, width=w * 0.9, color=RED, edgecolor="black", linewidth=0.4, hatch="////", zorder=3)
    ax.text(x030, last.val_pearson + 0.012, f"epoch {int(last.epoch)}\n1 seed", ha="center", va="bottom", fontsize=5)
    print(f"d disjoint 030 closure + fitness, epoch {int(last.epoch)} val {last.val_pearson:.3f} ({last.state})")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["random split\nover triples", "held-out\nquery pairs"])
    ax.set_xlim(-0.55, 1.75)
    ax.set_ylim(0, 0.8)
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.set_ylabel("held-out Pearson, trigenic interaction")
    ax.set_title("a no-interaction null reaches 0.40")
    handles = [Patch(facecolor=c, edgecolor="black", linewidth=0.4, label=l) for l, _, c in learners]
    handles.append(Patch(facecolor=RED, edgecolor="black", linewidth=0.4, hatch="////", label="CGT + closure + fitness (030)"))
    ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.02, 1.0), frameon=False, handlelength=1.0, handletextpad=0.3, borderpad=0.1, labelspacing=0.2)
    # The transformer's reading on that split differs from the baselines' (validation on one
    # split, epoch 29, three seeds, against five-fold test), so the note hangs under that group.
    ax.text(1.0, -0.22, "CGT: validation, one split", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=5, color="black")
    strip(ax)


def sweep() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    runs = pd.read_csv(osp.join(R025, "graph_reg_sweep_runs.csv"))
    hist = pd.read_csv(osp.join(R025, "graph_reg_sweep_history.csv"))
    summary = json.load(open(osp.join(R025, "graph_reg_sweep_summary.json")))
    return runs[runs.complete], hist[hist.run_id.isin(set(runs[runs.complete].run_id))], summary


def panel_e(ax: Axes, done: pd.DataFrame, summary: dict[str, Any]) -> None:
    pv = {p["arm"]: p["p_two_sided"] for p in summary["paired_vs_no_penalty"] if p["reading"] == "fixed"}
    arms = [("kl_0", "no\npenalty", GRAY), ("mask", "hard\nmask", ORANGE), ("kl_1", "soft\nprior", RED), ("random_1", "prior,\nrandom\ngraphs", PURPLE)]
    by: dict[str, pd.Series] = {}
    for x, (arm, _, color) in enumerate(arms):
        s = done[done.arm == arm].set_index("seed").val_pearson_fixed.dropna()
        by[arm] = s
        bars(ax, [x], [s.to_numpy()], [color])
        st = stars(pv.get(arm))
        if st:
            ax.text(x, s.max() + 0.012, st, ha="center", va="bottom", fontsize=6)
    shared = sorted(set(by["kl_1"].index) & set(by["random_1"].index))
    d = np.array([by["kl_1"][k] - by["random_1"][k] for k in shared])
    t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
    p = float(2 * stats.t.sf(abs(t), df=len(d) - 1))
    y = 0.474
    ax.plot([2, 2, 3, 3], [y - 0.004, y, y, y - 0.004], color="black", lw=0.5)
    ax.text(2.5, y + 0.002, f"+{d.mean():.3f} on every seed {stars(p)}", ha="center", va="bottom", fontsize=5)
    ax.set_xticks(range(4))
    ax.set_xticklabels([a[1] for a in arms])
    ax.set_ylim(0.38, 0.49)
    ax.yaxis.set_major_locator(MultipleLocator(0.02))
    ax.set_ylabel("validation Pearson, epoch 29")
    ax.set_title("a soft prior helps, a hard mask does not")
    strip(ax)


def panel_f(ax: Axes) -> None:
    d = pd.read_csv(osp.join(R025, "s3_closure_readout.csv"))
    pools = [("s0_ctrl_013", "S0\n\n377 k", GRAY), ("s0_fit_014", "S0 +\nfitness head\n377 k", YELLOW), ("s3_fit_031", "S3\nclosure\n1.12 M", RED)]
    for x, (arm, _, color) in enumerate(pools):
        vals = d[d.arm == arm]["val_gi_pearson_mean_ep10_29"].dropna().to_numpy(dtype=float)
        bars(ax, [x], [vals], [color])
    not_run(ax, 3, 0.383, "not run", RED)
    ax.set_xticks(range(4))
    ax.set_xticklabels([p[1] for p in pools] + ["030\nper entry\n13.5 M"], fontsize=5)
    ax.set_ylim(0.38, 0.49)
    ax.yaxis.set_major_locator(MultipleLocator(0.02))
    ax.set_ylabel("validation Pearson, epochs 10 to 29")
    ax.set_xlabel("training pool (records), same validation triples")
    ax.set_title("lower-order data adds 0.04")
    strip(ax)


LADDER = ["kl_0", "kl_1e-05", "kl_0.0001", "kl_0.001", "kl_0.01", "kl_0.1", "kl_1", "kl_10", "kl_100", "mask"]
LADDER_TICKS = ["−∞", "−5", "−4", "−3", "−2", "−1", "0", "1", "2", "mask"]


def ladder_color(arm: str) -> str:
    return GRAY if arm == "kl_0" else ORANGE if arm == "mask" else RED


def panel_g(ax: Axes, done: pd.DataFrame, summary: dict[str, Any]) -> None:
    pv = {p["arm"]: p["p_two_sided"] for p in summary["paired_vs_no_penalty"] if p["reading"] == "fixed"}
    for x, arm in enumerate(LADDER):
        vals = done[done.arm == arm].val_pearson_fixed.dropna().to_numpy(dtype=float)
        ax.scatter(x + np.linspace(-0.12, 0.12, len(vals)), vals, s=7, facecolor=ladder_color(arm), edgecolor="black", linewidth=0.4, zorder=4)
        ax.plot([x - 0.25, x + 0.25], [vals.mean()] * 2, color="black", lw=0.7, zorder=3)
        st = stars(pv.get(arm))
        if st:
            ax.text(x, 0.466, st, ha="center", va="bottom", fontsize=5.5)
    ax.set_xticks(range(len(LADDER)))
    ax.set_xticklabels(LADDER_TICKS, fontsize=5)
    ax.set_xlabel("log10 λ, the prior's weight")
    ax.set_ylim(0.40, 0.475)
    ax.yaxis.set_major_locator(MultipleLocator(0.02))
    ax.set_ylabel("validation Pearson, epoch 29")
    ax.set_title("rises to λ = 1 and holds to 100")
    ax.text(0.98, 0.03, "stars: paired t vs none\n* p < 0.05, ** 0.01, *** 0.001", transform=ax.transAxes, ha="right", va="bottom", fontsize=5, color="black")
    strip(ax)


def panel_h(ax: Axes, done: pd.DataFrame) -> None:
    pairs = [("kl_0.001", "random_0.001"), ("kl_0.1", "random_0.1"), ("kl_1", "random_1")]
    ax.axhline(0, color="black", lw=0.4, ls="--", zorder=1)
    for g, (bio, rnd) in enumerate(pairs):
        b = done[done.arm == bio].set_index("seed").val_pearson_fixed
        r = done[done.arm == rnd].set_index("seed").val_pearson_fixed
        seeds = sorted(set(b.index) & set(r.index))
        d = np.array([b[k] - r[k] for k in seeds])
        bars(ax, [g], [d], [RED])
        if len(d) > 1 and d.std(ddof=1) > 0:
            t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
            st = stars(float(2 * stats.t.sf(abs(t), df=len(d) - 1)))
            if st:
                ax.text(g, max(d.max(), d.mean()) + 0.004, st, ha="center", va="bottom", fontsize=6)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["λ = 0.001", "λ = 0.1", "λ = 1"])
    ax.set_ylim(-0.01, 0.035)
    ax.yaxis.set_major_locator(MultipleLocator(0.01))
    ax.set_ylabel("Pearson, biological − random graphs")
    ax.set_xlabel("same λ, same seed, epoch 29")
    ax.set_title("the biology is worth 0.017")
    strip(ax)


CURVES = [("random_1", "random graphs, λ = 1", PURPLE, "--"), ("kl_1", "soft prior, λ = 1", RED, "-"), ("mask", "hard mask", ORANGE, "-"), ("kl_0", "no penalty", GRAY, "-")]


def panel_i(ax: Axes, hist: pd.DataFrame) -> None:
    for z, (arm, label, color, ls) in enumerate(CURVES):
        curve(ax, hist, arm, "val/gene_interaction/Pearson", color, ls, label, 20 + z)
    ax.set_xlim(0, 29)
    ax.set_ylim(0.34, 0.47)
    ax.yaxis.set_major_locator(MultipleLocator(0.02))
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation Pearson")
    ax.set_title("the prior removes the late decline")
    ax.legend(loc="lower right", frameon=False, handlelength=1.6, handletextpad=0.4, borderpad=0.2, labelspacing=0.25)
    strip(ax)


def panel_j(ax: Axes, done: pd.DataFrame) -> None:
    arms = LADDER
    ticks = LADDER_TICKS
    for x, arm in enumerate(arms):
        vals = done[done.arm == arm].edge_recall_fixed.dropna().to_numpy(dtype=float)
        color = GRAY if arm == "kl_0" else RED
        if arm == "mask":
            vals = np.array([1.0])
        ax.scatter(x + np.linspace(-0.12, 0.12, len(vals)), vals, s=7, facecolor=color, edgecolor="black", linewidth=0.4, zorder=4)
    # The line runs straight down from the mask point, through no other point.
    ax.annotate("by construction", (len(arms) - 1, 1.0), xytext=(len(arms) - 1.15, 0.72), fontsize=5, ha="right", va="top", color="black",
                arrowprops={"arrowstyle": "-", "color": "black", "lw": 0.5, "shrinkA": 0, "shrinkB": 3})
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels(ticks, fontsize=5)
    ax.set_xlabel("log10 λ, the prior's weight")
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.set_ylabel("edge recall at degree")
    ax.set_title("every weight contracts attention")
    strip(ax)


def panel_k(ax: Axes, hist: pd.DataFrame) -> None:
    for z, (arm, label, color) in enumerate([("kl_1_60ep", "soft prior, λ = 1", RED), ("mask_60ep", "hard mask", ORANGE), ("kl_0_60ep", "no penalty", GRAY)]):
        curve(ax, hist, arm, "val/gene_interaction/Pearson", color, "-", label, 20 + z)
    ax.axvline(29, color="black", lw=0.4, ls=":")
    ax.text(29.8, 0.468, "30-epoch protocol", ha="left", va="top", fontsize=5, color="black")
    ax.set_xlim(0, 59)
    ax.set_ylim(0.34, 0.47)
    ax.yaxis.set_major_locator(MultipleLocator(0.02))
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation Pearson")
    ax.set_title("at 60 epochs only the prior holds")
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.0), frameon=False, handlelength=1.6, handletextpad=0.4, borderpad=0.2, labelspacing=0.25)
    strip(ax)


def panel_l(ax: Axes, hist: pd.DataFrame) -> None:
    for z, (arm, label, color) in enumerate([("kl_1", "gene table, soft prior", RED), ("kl_1_emb", "composite embedding, soft prior", YELLOW), ("kl_0_emb", "composite embedding, no penalty", GRAY)]):
        curve(ax, hist, arm, "val/gene_interaction/Pearson", color, "-", label, 20 + z)
    ax.set_xlim(0, 29)
    ax.set_ylim(-0.05, 0.47)
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation Pearson")
    ax.set_title("trains only with the prior")
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.4), frameon=False, handlelength=1.6, handletextpad=0.4, borderpad=0.2, labelspacing=0.25)
    strip(ax)


# ------------------------------------------------------- row 4: the wild-type genes
WT = osp.join(R025, "wt_gene_embeddings")
CLUSTER_COLORS = [ORANGE, RED, PURPLE, YELLOW, BLUE]  # the five best-enriched clusters


def wt_data() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    # The maps draw the t-SNE at perplexity 30, exaggeration 4 (tsne_coords.csv, from
    # wt_gene_embedding_tsne.py; review 2026-10-13), which gives the trained cloud the
    # panel; the UMAP columns stay in cluster_assignments.csv.
    assign = pd.read_csv(osp.join(WT, "cluster_assignments.csv")).merge(pd.read_csv(osp.join(WT, "tsne_coords.csv")), on="gene", how="left")
    assert assign.encoder_tsne_1.notna().all()
    enr = pd.read_csv(osp.join(WT, "go_enrichment.csv"))
    summary = json.load(open(osp.join(WT, "summary.json")))
    return assign, enr, summary


def best_terms(enr: pd.DataFrame, rep: str) -> pd.DataFrame:
    """One row per cluster: its best term by q, ordered by q."""
    sub = enr[enr.representation == rep].sort_values("q")
    return sub.groupby("cluster", as_index=False).first().sort_values("q")


def umap_by_cluster(ax: Axes, assign: pd.DataFrame, enr: pd.DataFrame, rep: str, title: str) -> None:
    best = best_terms(enr, rep)
    top = [int(c) for c in best[best.q < 0.05].cluster.head(len(CLUSTER_COLORS))]
    color_of = {c: CLUSTER_COLORS[i] for i, c in enumerate(top)}
    x, y, lab = assign[f"{rep}_{MAP}_1"], assign[f"{rep}_{MAP}_2"], assign[f"{rep}_cluster"]
    other = ~lab.isin(top)
    ax.scatter(x[other], y[other], s=1.2, color="#C8C8C8", linewidths=0, rasterized=True, zorder=2)
    for c in top:
        m = lab == c
        ax.scatter(x[m], y[m], s=1.2, color=color_of[c], linewidths=0, rasterized=True, zorder=3)
        ax.text(x[m].median(), y[m].median(), str(c), fontsize=6, fontweight="bold", ha="center", va="center", zorder=5)
    ax.margins(0.02)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(title)
    ax.text(0.0, -0.06, f"{MAP_NAME} of 6,607 genes", transform=ax.transAxes, fontsize=5, color="black", ha="left", va="bottom")


def panel_m(ax: Axes, assign: pd.DataFrame, enr: pd.DataFrame, summary: dict[str, Any]) -> None:
    rep = next(r for r in summary["representations"] if r["name"] == "encoder")
    umap_by_cluster(ax, assign, enr, "encoder", f"encoder output: {rep['n_clusters']} clusters, {rep['n_enriched_clusters']} GO-enriched")


def panel_n(ax: Axes, enr: pd.DataFrame) -> None:
    best = best_terms(enr, "encoder")
    top = [int(c) for c in best[best.q < 0.05].cluster.head(len(CLUSTER_COLORS))]
    color_of = {c: CLUSTER_COLORS[i] for i, c in enumerate(top)}
    ys = np.arange(len(best))[::-1]
    vals = [-np.log10(max(r.q, 1e-300)) for _, r in best.iterrows()]
    xmax = max(vals) * 1.75
    for yy, v, (_, r) in zip(ys, vals, best.iterrows()):
        c = int(r.cluster)
        ax.barh(yy, v, color=color_of.get(c, "#C8C8C8"), edgecolor="black", linewidth=0.4, height=0.7, zorder=3)
        # The term name sits at the end of its bar; the tick label is the cluster number,
        # which is what the map in n carries, so nothing hangs left into n.
        name = r.go_name if len(r.go_name) <= 26 else r.go_name[:25] + "\u2026"
        ax.text(v + xmax * 0.015, yy, name, fontsize=5, ha="left", va="center", zorder=4)
    ax.set_yticks(ys)
    ax.set_yticklabels([str(int(r.cluster)) for _, r in best.iterrows()], fontsize=5)
    ax.set_xlim(0, xmax)
    ax.axvline(-np.log10(0.05), color="black", lw=0.5, ls="--")
    ax.set_xlabel("best GO term per cluster, \u2212log10 q (BH)")
    ax.set_ylabel("cluster")
    ax.set_title("9 of 12 clusters carry a term at q < 0.05")
    strip(ax, grid="x")


def panel_o(ax: Axes, assign: pd.DataFrame) -> None:
    x, y, it = assign[f"encoder_{MAP}_1"], assign[f"encoder_{MAP}_2"], assign.in_train_triple.astype(bool)
    ax.scatter(x[~it], y[~it], s=1.2, color="#C8C8C8", linewidths=0, rasterized=True, zorder=2)
    ax.scatter(x[it], y[it], s=1.2, color=RED, linewidths=0, rasterized=True, zorder=3)
    ax.margins(0.02)
    lo, hi = float(y.min()), float(y.max())
    ax.set_ylim(lo - 0.02 * (hi - lo), hi + 0.30 * (hi - lo))
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    n_in = int(it.sum())
    ax.set_title("the slivers: genes in no training triple")
    ax.legend(
        handles=[
            Patch(facecolor=RED, edgecolor="none", label=f"in a training triple, {n_in:,}"),
            Patch(facecolor="#C8C8C8", edgecolor="none", label=f"in none, {len(it) - n_in:,}"),
        ],
        loc="upper right", frameon=False, handlelength=1.0, handletextpad=0.4, borderpad=0.2, labelspacing=0.25,
    )
    ax.text(0.0, -0.06, f"{MAP_NAME} of 6,607 genes", transform=ax.transAxes, fontsize=5, color="black", ha="left", va="bottom")


def panel_p(ax: Axes, assign: pd.DataFrame, enr: pd.DataFrame, summary: dict[str, Any]) -> None:
    rep = next(r for r in summary["representations"] if r["name"] == "table")
    umap_by_cluster(ax, assign, enr, "table", f"input table: {rep['n_clusters']} clusters, {rep['n_enriched_clusters']} enriched")


# ----------------------------------------------------------------------------- sheet


def main() -> None:
    done, hist, summary = sweep()
    assign, enr, wt = wt_data()
    fig, axes = plt.subplots(4, 4, figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(200)))
    fig.subplots_adjust(left=0.055, right=0.99, bottom=0.085, top=0.965, wspace=0.34, hspace=0.62)
    ax = axes.ravel()
    # Rows 2 to 4 drop by a strip that holds the color key under row 1, beside d, e, f.
    key_strip = 0.022
    for a_ in ax[4:]:
        pos = a_.get_position()
        a_.set_position([pos.x0, pos.y0 - key_strip, pos.width, pos.height])
    panel_a(ax[0])
    panel_b(ax[1])
    panel_c(ax[2])
    panel_d(ax[3])
    panel_e(ax[4], done, summary)
    panel_f(ax[5])
    panel_g(ax[6], done, summary)
    panel_h(ax[7], done)
    panel_i(ax[8], hist)
    panel_j(ax[9], done)
    panel_k(ax[10], hist)
    panel_l(ax[11], hist)
    # m coverage map, n cluster map, o terms, p table: the cluster colors sit beside the
    # term bars that carry them (review 2026-10-13).
    panel_o(ax[12], assign)
    panel_m(ax[13], assign, enr, wt)
    panel_n(ax[14], enr)
    panel_p(ax[15], assign, enr, wt)
    letters_status(fig, list(ax), ["ready"] * 16)
    y_key = (ax[0].get_position().y0 + ax[4].get_position().y1) / 2 - 0.004
    keys(fig, y_key)
    fig.subplots_adjust(bottom=0.04)
    os.makedirs(IMG_DIR, exist_ok=True)
    stem = osp.join(IMG_DIR, "fig2_gene_interactions")
    savefig_true_size_svg(fig, stem + ".svg")
    fig.savefig(stem + ".png", dpi=300)
    fig.savefig(stem + ".pdf")
    savefig_true_size_svg(fig, f"{stem}_{timestamp()}.svg")
    print("wrote", stem + ".svg")


if __name__ == "__main__":
    main()
