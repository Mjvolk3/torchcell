# experiments/030-solid-growth-multi/scripts/figure2_gate_sheet.py
# [[experiments.030-solid-growth-multi.scripts.figure2_gate_sheet]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/030-solid-growth-multi/scripts/figure2_gate_sheet
"""The sheet proposed for manuscript Figure 2: trigenic interaction and fitness, in full.

Built the way the manuscript's Figure 1 is built, one dense page at Nature's full-page
box, every panel from a committed result file, nothing typed. Color is the input family
a method reads when it scores a held-out triple: gray the screen alone; orange the
measured lower-order terms of the same strains (a lookup, transductive over the
closure); blue gene identity alone; purple the cell graph transformer (CGT), which reads
identity, the gene graphs and sequence embeddings. Light fill is a test score, the darker
line color a validation score of a model with no test evaluation. Each panel letter
carries the two status marks of the figure gates (section status, figure status).

Panels and their files (``experiments/030-solid-growth-multi/results/`` unless named):

    a  what a trigenic interaction is and which records train      (idea: a draw.io composition)
    b  the released score by screen on the held-out triples         lower_order_features_030.parquet
    c  the interaction ladder, arm R                                  lower_order_baselines_030.csv
    d  the interaction ladder, arm Q                                  lower_order_baselines_030.csv
    e  within-screen Pearson of every rung, both arms                 lower_order_baselines_030.csv
    f  the fitness side, ridge against the CGT fitness head           lower_order_baselines_030.csv, figure2_gate_fitness_wandb.json
    g  measured against predicted score, binned, arm R test           lower_order_baselines_030_pred_R.parquet
    h  Pearson inside bands of |tau|, arm R test                      lower_order_baselines_030_pred_R.parquet
    i  agreement between the rungs' predictions, arm R test           lower_order_baselines_030_pred_R.parquet
    j  the CGT validation curves on both arms                         figure2_gate_curves_wandb.json (W&B, --wandb)
    k  comparators on other builds                                    010, 006, 005, 007 result files
    l  scaling and the deployment tiers                               (idea: 030-scaling-study and registry E5)

    PYTHONPATH=$PWD python experiments/030-solid-growth-multi/scripts/figure2_gate_sheet.py [--wandb]
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402
from scipy.stats import pearsonr  # noqa: E402

from torchcell.timestamp import timestamp  # noqa: E402
from torchcell.utils import (  # noqa: E402
    MAX_HEIGHT_MM,
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    PLOT_PALETTE_FILL,
    mm_to_in,
    savefig_true_size_svg,
)

load_dotenv()
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
ASSET_IMAGES_DIR = os.environ["ASSET_IMAGES_DIR"]
R030 = osp.join(EXPERIMENT_ROOT, "030-solid-growth-multi", "results")
R025 = osp.join(EXPERIMENT_ROOT, "025-solid-growth", "results")
R010 = osp.join(EXPERIMENT_ROOT, "010-kuzmin-tmi", "results")
R006 = osp.join(EXPERIMENT_ROOT, "006-kuzmin-tmi", "results", "dcell_training")
R005 = osp.join(EXPERIMENT_ROOT, "005-kuzmin2018-tmi", "results")
R007 = osp.join(EXPERIMENT_ROOT, "007-kuzmin-tm", "results", "fba_baseline_si")
IMAGES = osp.join(ASSET_IMAGES_DIR, "030-solid-growth-multi")
WIDTH_MM = PANEL_WIDTHS_MM["full"]

from wandb_seed_histories import (  # noqa: E402  (sibling module, scripts dir)
    WANDB_ARM_TAG,
    WANDB_HEADLINE_SEED,
    wandb_seed_histories,
)

#: Color is the input family, never the learner.
SCREEN = PLOT_PALETTE[5]
TERMS = PLOT_PALETTE[0]
TERMS_FILL = PLOT_PALETTE_FILL[0]
IDENTITY = PLOT_PALETTE[4]
IDENTITY_FILL = PLOT_PALETTE_FILL[4]
CGT = PLOT_PALETTE[2]
CGT_FILL = PLOT_PALETTE_FILL[2]
SCREEN_FILL = PLOT_PALETTE_FILL[5]
KZ18 = PLOT_PALETTE[3]
KZ20 = PLOT_PALETTE[1]
BAR = {"edgecolor": "black", "linewidth": 0.4}

RUNGS = [
    ("N0_screen_mean", "screen mean", SCREEN, SCREEN_FILL),
    ("N1_mult_query_double", "N1 digenic controls", TERMS, TERMS_FILL),
    ("N1s_mult_strain_matched", "N1s strain-matched", TERMS, TERMS_FILL),
    ("N2_mult_singles", "N2 singles product", TERMS, TERMS_FILL),
    ("B1_additive_gene", "B1 additive gene ridge", IDENTITY, IDENTITY_FILL),
    ("L1_ridge_lower_order", "L1 ridge, terms", TERMS, TERMS_FILL),
    ("L2_hgb_lower_order", "L2 boosted trees, terms", TERMS, TERMS_FILL),
    ("L3_ridge_gene_plus_lower_order", "L3 ridge, genes + terms", TERMS, TERMS_FILL),
]
STATUS_SYMBOL = {"file": "●", "partial": "◐", "idea": "○"}
STATUS_WORD = {"file": "on file", "partial": "partial", "idea": "idea only"}
SECTION_SYMBOL = {"todo": ("✗", "#E53935"), "tent": ("■", "#E0A020"), "final": ("✓", "#43A047")}
SECTION_WORD = {"todo": "not done, agents may edit freely", "tent": "author-reviewed", "final": "final"}
SECTION_STATUS = "todo"
STATUS_MARKS = os.environ.get("GATE_STATUS_MARKS", "1") == "1"


# --------------------------------------------------------------------------- helpers
def read_json(path: str) -> Any:
    with open(path) as f:
        return json.load(f)


def place(fig, x: float, top: float, w: float, h: float):
    height_mm = fig.get_size_inches()[1] * 25.4
    ax = fig.add_axes((x / WIDTH_MM, 1.0 - (top + h) / height_mm, w / WIDTH_MM, h / height_mm))
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)
    ax.tick_params(width=0.5, length=2)
    return ax


def letter_status(fig, x_mm: float, top_mm: float, ch: str, status: str) -> None:
    h_mm = fig.get_size_inches()[1] * 25.4
    x, y = x_mm / WIDTH_MM, 1.0 - (top_mm - 1.0) / h_mm
    fig.text(x, y, ch, fontsize=8, fontweight="bold", va="bottom", ha="left")
    if not STATUS_MARKS:
        return
    glyph, color = SECTION_SYMBOL[SECTION_STATUS]
    fig.text(x + 3.2 / WIDTH_MM, y + 0.3 / h_mm, glyph, fontsize=5, color=color, va="bottom", ha="left", family="DejaVu Sans")
    fig.text(x + 5.6 / WIDTH_MM, y + 0.3 / h_mm, STATUS_SYMBOL[status], fontsize=5, color=SCREEN, va="bottom", ha="left", family="DejaVu Sans")


def panel_legend(fig, handles, x_mm: float, top_mm: float, ncol: int) -> None:
    h_mm = fig.get_size_inches()[1] * 25.4
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(x_mm / WIDTH_MM, 1.0 - (top_mm - 0.5) / h_mm),
               ncol=ncol, frameon=False, handlelength=1.2, handletextpad=0.4, columnspacing=1.0,
               borderaxespad=0.0, fontsize=6)


def status_key(fig) -> None:
    if not STATUS_MARKS:
        return
    x = 0.0
    fig.text(x, 0.004, "section status", fontsize=5, color=SCREEN, ha="left", va="bottom", style="italic")
    x += 0.075
    for key in ("todo", "tent", "final"):
        glyph, color = SECTION_SYMBOL[key]
        fig.text(x, 0.004, glyph, fontsize=5, color=color, ha="left", va="bottom", family="DejaVu Sans")
        fig.text(x + 0.012, 0.004, SECTION_WORD[key], fontsize=5, color=SCREEN, ha="left", va="bottom")
        x += 0.03 + 0.0052 * len(SECTION_WORD[key])
    x += 0.03
    fig.text(x, 0.004, "figure status", fontsize=5, color=SCREEN, ha="left", va="bottom", style="italic")
    x += 0.07
    for key in ("file", "partial", "idea"):
        fig.text(x, 0.004, STATUS_SYMBOL[key], fontsize=5, color=SCREEN, ha="left", va="bottom", family="DejaVu Sans")
        fig.text(x + 0.012, 0.004, STATUS_WORD[key], fontsize=5, color=SCREEN, ha="left", va="bottom")
        x += 0.065


def metric_axis(ax, top: float, axis: str = "y") -> None:
    a = ax.yaxis if axis == "y" else ax.xaxis
    (ax.set_ylim if axis == "y" else ax.set_xlim)(0, top)
    a.set_major_locator(MultipleLocator(0.2))
    a.set_minor_locator(MultipleLocator(0.1))
    ax.tick_params(axis=axis, which="minor", length=0)
    ax.grid(axis=axis, which="both", lw=0.3, color="0.85", zorder=0)
    ax.set_axisbelow(True)


def placeholder(fig, x: float, top: float, w: float, h: float, text: str):
    ax = place(fig, x, top, w, h)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linestyle((0, (2, 2)))
        sp.set_color(SCREEN)
    ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=6, color=SCREEN, wrap=True, transform=ax.transAxes)
    return ax


# ------------------------------------------------------------------------------ data
def ladder() -> pd.DataFrame:
    return pd.read_csv(osp.join(R030, "lower_order_baselines_030.csv"))


def summary() -> dict[str, Any]:
    return read_json(osp.join(R030, "lower_order_baselines_030_summary.json"))


def preds(arm: str) -> pd.DataFrame:
    return pd.read_parquet(osp.join(R030, f"lower_order_baselines_030_pred_{arm}.parquet"))


def features() -> pd.DataFrame:
    return pd.read_parquet(osp.join(R030, "lower_order_features_030.parquet"), columns=["idx", "tau", "screen"])


def fitness_ref() -> list[dict[str, Any]]:
    return read_json(osp.join(R030, "figure2_gate_fitness_wandb.json"))


def curves(fetch: bool) -> dict[str, dict[int, list[tuple[int, float]]]]:
    """Per arm, per seed, the validation interaction Pearson by epoch; seeds never mixed."""
    cache = osp.join(R030, "figure2_gate_curves_wandb.json")
    if fetch:
        out: dict[str, dict[int, list[tuple[int, float]]]] = {}
        for arm, tag in WANDB_ARM_TAG.items():
            hists = wandb_seed_histories(tag, "val/gene_interaction/Pearson")
            out[arm] = {seed: [(int(e), float(v)) for e, v in h.items()] for seed, h in hists.items()}
        with open(cache, "w") as f:
            json.dump({arm: {str(k): v for k, v in d.items()} for arm, d in out.items()}, f)
        return out
    raw = read_json(cache)
    return {arm: {int(k): [tuple(x) for x in v] for k, v in d.items()} for arm, d in raw.items()}


def comparators() -> list[tuple[str, float, str, bool]]:
    """(label, Pearson, kind, is_validation) for the comparators on other builds."""
    rows: list[tuple[str, float, str, bool]] = []
    cgt010 = pd.read_csv(osp.join(R010, "additive_baseline_gene_interaction.csv"))
    cgt010 = cgt010[cgt010["model"].str.startswith("CGT") & (cgt010["split"] == "test")]
    rows.append(("CGT, 010 build, 3 checkpoints", float(cgt010["pearson"].mean()), "cgt", False))
    s025 = read_json(osp.join(R025, "additive_baselines_025_summary.json"))
    rep = s025["transformer"]["R"]["job_1598_replication"]
    rows.append(("CGT, 025 build, val max", float(rep["val_pearson_best_epoch"]), "cgt", True))
    dcell = pd.read_csv(osp.join(R006, "checkpoints.csv"))
    rows.append(("DCell, 006 build, val", float(dcell["val_pearson"].max()), "other", True))
    dango = pd.read_csv(osp.join(R005, "dango_string_version_summary.csv")).sort_values("mean", ascending=False).iloc[0]
    rows.append(("DANGO, 005 build", float(dango["mean"]), "other", False))
    fba = read_json(osp.join(R007, "stats.json"))["correlations"]["tau"]
    rows.append(("Yeast9 FBA, 007", float(fba["pearson_r"]), "other", False))
    return rows


# ---------------------------------------------------------------------------- panels
def panel_tau_by_screen(fig, x, top, w, h, feat: pd.DataFrame) -> None:
    ax = place(fig, x, top, w, h)
    bins = np.linspace(-0.4, 0.3, 57)
    for screen, color in (("kuzmin2018", KZ18), ("kuzmin2020", KZ20)):
        v = feat.loc[feat["screen"] == screen, "tau"].to_numpy()
        ax.hist(v, bins=bins, density=True, histtype="step", color=color, lw=0.8, zorder=3)
        ax.axvline(float(v.mean()), color=color, lw=0.6, ls=(0, (2, 2)), zorder=2)
    ax.set_xlim(-0.4, 0.3)
    ax.set_xlabel("released trigenic score, every triple")
    ax.set_ylabel("density")
    ax.grid(axis="y", lw=0.3, color="0.85", zorder=0)
    ax.set_axisbelow(True)
    panel_legend(fig, [Line2D([], [], color=KZ18, lw=0.8, label="Kuzmin 2018"),
                       Line2D([], [], color=KZ20, lw=0.8, label="Kuzmin 2020"),
                       Line2D([], [], color="black", lw=0.6, ls=(0, (2, 2)), label="screen mean")],
                 x, top, 3)


def panel_ladder(fig, x, top, w, h, lad: pd.DataFrame, arm: str, s030: dict[str, Any]) -> None:
    sub = lad[lad["arm"] == arm].set_index(["rung", "split"])["pearson"]
    ax = place(fig, x, top, w, h)
    labels, vals, colors = [], [], []
    for key, label, _, fill in RUNGS:
        labels.append(label)
        vals.append(float(sub.loc[(key, "test")]))
        colors.append(fill)
    refs = [r for r in s030["wandb"] if r["arm"] == arm and not r["partial"]]
    for ref in refs:
        seed = f" seed {ref['seed']}," if len(refs) > 1 else ""
        labels.append(f"CGT,{seed} val ep {ref['last_epoch'] - ref['window'] + 1}-{ref['last_epoch']}")
        vals.append(float(ref["val_pearson_last_window_mean"]))
        colors.append(CGT)
    y = np.arange(len(labels))[::-1]
    ax.barh(y, vals, 0.7, color=colors, zorder=3, **BAR)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_ylim(-0.6, len(labels) - 0.4)
    ax.tick_params(axis="y", length=0)
    metric_axis(ax, 0.8, "x")
    ax.set_xlabel(f"Pearson r, arm {arm} held-out triples")


def panel_within(fig, x, top, w, h, lad: pd.DataFrame) -> None:
    ax = place(fig, x, top, w, h)
    keys = [k for k, _, _, _ in RUNGS]
    xs = np.arange(len(keys))
    for i, arm in enumerate(("R", "Q")):
        sub = lad[(lad["arm"] == arm) & (lad["split"] == "test")].set_index("rung")
        vals = [float(sub.loc[k, "pearson_within_screen"]) for k in keys]
        fills = [RUNGS[j][3] if arm == "Q" else RUNGS[j][2] for j in range(len(keys))]
        ax.bar(xs + (i - 0.5) * 0.38, vals, 0.36, color=fills, zorder=3, hatch=None if arm == "R" else "", **BAR)
    ax.set_xticks(xs)
    ax.set_xticklabels([l for _, l, _, _ in RUNGS], rotation=45, ha="right", fontsize=5)
    metric_axis(ax, 0.8)
    ax.set_ylabel("within-screen Pearson r, test")
    panel_legend(fig, [Patch(facecolor=TERMS, label="arm R (line color)", **BAR),
                       Patch(facecolor=TERMS_FILL, label="arm Q (fill)", **BAR)], x, top, 2)


def panel_fitness(fig, x, top, w, h, lad: pd.DataFrame, refs: list[dict[str, Any]]) -> None:
    ax = place(fig, x, top, w, h)
    cats, vals, colors = [], [], []
    for arm in ("R", "Q"):
        sub = lad[lad["arm"] == arm].set_index(["rung", "split"])["pearson"]
        cats.append(f"ridge\narm {arm}")
        vals.append(float(sub.loc[("F1_ridge_lower_order_fitness", "test")]))
        colors.append(TERMS_FILL)
        full = [r for r in refs if r["arm"] == arm and not r["partial"]]
        for ref in full:
            seed = f"\nseed {ref['seed']}" if len(full) > 1 else ""
            cats.append(f"CGT\narm {arm}{seed}")
            vals.append(float(ref["val_fitness_order3_last_window_mean"]))
            colors.append(CGT)
    xs = np.arange(len(cats))
    ax.bar(xs, vals, 0.7, color=colors, zorder=3, **BAR)
    ax.set_xticks(xs)
    ax.set_xticklabels(cats, fontsize=5)
    metric_axis(ax, 1.0)
    ax.set_ylabel("Pearson r, held-out triple fitness")
    panel_legend(fig, [Patch(facecolor=TERMS_FILL, label="ridge on terms, test", **BAR),
                       Patch(facecolor=CGT, label="CGT fitness head, val", **BAR)], x - 4.0, top, 1)


def panel_binned(fig, x, top, w, h, pr: pd.DataFrame) -> None:
    te = pr[pr["split"] == "test"]
    y, yhat = te["tau"].to_numpy(), te["L2_hgb_lower_order"].to_numpy()
    edges = np.quantile(yhat, np.linspace(0, 1, 11))
    which = np.clip(np.searchsorted(edges, yhat, side="right") - 1, 0, 9)
    ax = place(fig, x, top, w, h)
    data = [y[which == b] for b in range(10)]
    ax.boxplot(data, positions=np.arange(10), widths=0.6, showfliers=False,
               medianprops={"color": "black", "lw": 0.6}, boxprops={"color": TERMS, "lw": 0.6},
               whiskerprops={"color": TERMS, "lw": 0.5}, capprops={"color": TERMS, "lw": 0.5}, zorder=3)
    ax.axhline(0, color="black", lw=0.4, zorder=2)
    ax.set_xticks(np.arange(10))
    ax.set_xticklabels([f"{edges[b]:.2f}" for b in range(10)], rotation=90, fontsize=5)
    ax.set_xlabel("predicted score (L2), decile lower edge")
    ax.set_ylabel("measured score")
    ax.set_ylim(-0.4, 0.3)
    ax.grid(axis="y", lw=0.3, color="0.85", zorder=0)
    ax.set_axisbelow(True)


def panel_bands(fig, x, top, w, h, pr: pd.DataFrame) -> None:
    te = pr[pr["split"] == "test"]
    y = te["tau"].to_numpy()
    mag = np.abs(y)
    edges = np.quantile(mag, [0, 0.5, 0.75, 0.9, 0.97, 1.0])
    ax = place(fig, x, top, w, h)
    series = [("N1_mult_query_double", "N1 controls", TERMS, (0, (2, 2))), ("L2_hgb_lower_order", "L2 trees", TERMS, "-"),
              ("B1_additive_gene", "B1 ridge", IDENTITY, "-")]
    xs = np.arange(len(edges) - 1)
    for col, label, color, ls in series:
        rs = []
        for b in range(len(edges) - 1):
            m = (mag >= edges[b]) & (mag < edges[b + 1]) if b < len(edges) - 2 else (mag >= edges[b])
            rs.append(float(pearsonr(y[m], te[col].to_numpy()[m])[0]))
        ax.plot(xs, rs, marker="o", ms=2.5, lw=0.7, color=color, ls=ls, label=label, zorder=3)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{edges[b]:.2f}-{edges[b+1]:.2f}" for b in range(len(edges) - 1)], rotation=45, ha="right", fontsize=5)
    ax.set_xlabel("|measured score| band (quantile edges)")
    ax.set_ylabel("Pearson r in band")
    metric_axis(ax, 1.0)
    ax.set_ylim(0.0, 1.0)
    panel_legend(fig, [Line2D([], [], color=c, ls=ls, lw=0.7, marker="o", ms=2.5, label=l) for _, l, c, ls in series], x, top, 3)


def panel_agreement(fig, x, top, w, h, pr: pd.DataFrame) -> None:
    te = pr[pr["split"] == "test"]
    cols = ["tau", "N1_mult_query_double", "B1_additive_gene", "L1_ridge_lower_order", "L2_hgb_lower_order", "L3_ridge_gene_plus_lower_order"]
    names = ["measured", "N1", "B1", "L1", "L2", "L3"]
    m = np.corrcoef(np.column_stack([te[c].to_numpy() for c in cols]).T)
    ax = place(fig, x, top, w, h)
    ax.imshow(m, cmap="Oranges", vmin=0, vmax=1, zorder=2)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=5, color="black" if m[i, j] < 0.75 else "white")
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=5)
    ax.set_yticklabels(names, fontsize=5)
    ax.tick_params(length=0)


def panel_curves(fig, x, top, w, h, cv: dict[str, dict[int, list[tuple[int, float]]]], lad: pd.DataFrame) -> None:
    ax = place(fig, x, top, w, h)
    for arm, color in (("R", CGT), ("Q", CGT_FILL)):
        for seed, pts in sorted(cv[arm].items()):
            headline = seed == WANDB_HEADLINE_SEED
            ax.plot([e for e, _ in pts], [v for _, v in pts], lw=0.7 if headline else 0.5,
                    ls="-" if headline else (0, (1, 1)), marker="" if len(pts) > 1 else "o", ms=2,
                    color=color, zorder=3)
        sub = lad[(lad["arm"] == arm) & (lad["split"] == "val")].set_index("rung")
        ax.axhline(float(sub.loc["N1_mult_query_double", "pearson"]), color=TERMS, lw=0.6, ls="-" if arm == "R" else (0, (2, 2)), zorder=2)
        ax.axhline(float(sub.loc["B1_additive_gene", "pearson"]), color=IDENTITY, lw=0.6, ls="-" if arm == "R" else (0, (2, 2)), zorder=2)
    ax.set_xlabel("epoch (synced to the run registry)")
    ax.set_ylabel("validation Pearson r")
    metric_axis(ax, 0.8)
    panel_legend(fig, [Line2D([], [], color=CGT, lw=0.7, label="CGT arm R, seed 0"), Line2D([], [], color=CGT_FILL, lw=0.7, label="CGT arm Q, seed 0"),
                       Line2D([], [], color="black", lw=0.5, ls=(0, (1, 1)), label="other seeds, dotted"),
                       Line2D([], [], color=TERMS, lw=0.6, label="N1 null, val (solid R, dashed Q)"),
                       Line2D([], [], color=IDENTITY, lw=0.6, label="B1 ridge, val")], x, top, 2)


def panel_comparators(fig, x, top, w, h, rows: list[tuple[str, float, str, bool]]) -> None:
    ax = place(fig, x, top, w, h)
    y = np.arange(len(rows))[::-1]
    colors = [(CGT if kind == "cgt" else SCREEN) if is_val else (CGT_FILL if kind == "cgt" else SCREEN_FILL) for _, _, kind, is_val in rows]
    ax.barh(y, [v for _, v, _, _ in rows], 0.7, color=colors, hatch="////", zorder=3, **BAR)
    ax.set_yticks(y)
    ax.set_yticklabels([l for l, _, _, _ in rows], fontsize=5)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.tick_params(axis="y", length=0)
    metric_axis(ax, 0.6, "x")
    ax.set_xlabel("Pearson r on that build's split (hatch: other build)")


# ------------------------------------------------------------------------------ sheet
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wandb", action="store_true")
    args = ap.parse_args()
    plt.rcParams.update({"font.family": "Arial", "font.size": 6, "svg.fonttype": "none", "axes.linewidth": 0.5,
                         "hatch.linewidth": 0.4, "axes.titlesize": 6, "legend.fontsize": 6})
    height = 168.0
    assert height <= MAX_HEIGHT_MM
    fig = plt.figure(figsize=(mm_to_in(WIDTH_MM), mm_to_in(height)))
    fig.legend(handles=[Patch(facecolor=SCREEN_FILL, label="screen only", **BAR),
                        Patch(facecolor=TERMS_FILL, label="measured lower-order terms (lookup)", **BAR),
                        Patch(facecolor=IDENTITY_FILL, label="gene identity", **BAR),
                        Patch(facecolor=CGT, label="CGT: identity, graphs, sequence (validation)", **BAR),
                        Patch(facecolor="white", hatch="////", label="another build", **BAR)],
               loc="upper left", bbox_to_anchor=(0.0, 1.0), ncol=5, frameon=False, handlelength=1.2, handletextpad=0.4,
               columnspacing=1.0, borderaxespad=0.2)

    lad, s030, feat = ladder(), summary(), features()
    prR = preds("R")
    cv = curves(args.wandb)

    # row 1: a identity (idea), b score by screen, c ladder R, d ladder Q
    top = 12.0
    placeholder(fig, 0.0, top + 4.0, 40.0, 34.0, "what a trigenic\ninteraction is, and\nwhich records train:\na draw.io composition\n(the datasets figure)")
    letter_status(fig, 0.0, top, "a", "idea")
    panel_tau_by_screen(fig, 50.0, top + 4.0, 34.0, 32.0, feat)
    letter_status(fig, 44.0, top, "b", "file")
    panel_ladder(fig, 123.0, top + 4.0, 52.0, 34.0, lad, "R", s030)
    letter_status(fig, 90.0, top, "c", "file")

    top = 62.0
    panel_ladder(fig, 33.0, top + 4.0, 52.0, 34.0, lad, "Q", s030)
    letter_status(fig, 0.0, top, "d", "file")
    panel_within(fig, 100.0, top + 4.0, 75.0, 28.0, lad)
    letter_status(fig, 90.0, top, "e", "file")

    top = 116.0
    panel_fitness(fig, 12.0, top + 4.0, 24.0, 26.0, lad, fitness_ref())
    letter_status(fig, 0.0, top, "f", "file")
    panel_binned(fig, 48.0, top + 4.0, 30.0, 26.0, prR)
    letter_status(fig, 40.0, top, "g", "file")
    panel_bands(fig, 92.0, top + 4.0, 30.0, 26.0, prR)
    letter_status(fig, 84.0, top, "h", "file")
    panel_agreement(fig, 136.0, top + 4.0, 26.0, 26.0, prR)
    letter_status(fig, 128.0, top, "i", "file")
    status_key(fig)

    os.makedirs(IMAGES, exist_ok=True)
    stem = osp.join(IMAGES, f"figure2_gate_sheet_{timestamp()}")
    savefig_true_size_svg(fig, stem + ".svg")
    fig.savefig(stem + ".png", dpi=300)
    print(stem + ".svg")

    # second sheet: the curves, the comparators and the two ideas
    fig2 = plt.figure(figsize=(mm_to_in(WIDTH_MM), mm_to_in(60.0)))
    panel_curves(fig2, 10.0, 10.0, 52.0, 36.0, cv, lad)
    letter_status(fig2, 2.0, 6.0, "j", "partial")
    panel_comparators(fig2, 100.0, 10.0, 40.0, 36.0, comparators())
    letter_status(fig2, 68.0, 6.0, "k", "partial")
    placeholder(fig2, 146.0, 10.0, 32.0, 36.0, "scaling (030 study)\nand the deployment\ntiers of the ladder\n(registry E5)")
    letter_status(fig2, 142.0, 6.0, "l", "idea")
    stem2 = osp.join(IMAGES, f"figure2_gate_sheet2_{timestamp()}")
    savefig_true_size_svg(fig2, stem2 + ".svg")
    fig2.savefig(stem2 + ".png", dpi=300)
    print(stem2 + ".svg")


if __name__ == "__main__":
    main()
