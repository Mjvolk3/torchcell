# experiments/028-gene-essentiality/scripts/summary_figure.py
# [[experiments.028-gene-essentiality.scripts.summary_figure]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/028-gene-essentiality/scripts/summary_figure

"""The study's conclusions on one page: which method, which module form, what scales,
and which genes are still missed.

Seven panels, each answering one question from the committed result files:

- **a** what ranks the 195 released held-out genes best, every method on one axis;
- **b**, **c** which form of the metabolic module helps, on one backbone, and what each
  costs per cell;
- **d** what transfers to the 4,674 genome-wide genes no model saw a label for;
- **e** what the winning module form costs over the whole solid-growth build;
- **f**, **g** what the misranked genes of each held-out set share
  (``misclass_enrichment.py``), under the best model of each family.

One color means one thing in every panel: gray a comparator with nothing of ours learned,
yellow a learned model that reads no solve and no graph, orange the transformer reading
the metabolic solve, red message passing over the gene graphs, purple an average of the
two families. A hatch marks a linear program solved inside the training loop.

Panels are placed in millimeters, so a label cannot drift onto its neighbor when a panel
is added. Nothing is typed here: every number is read from ``results/``.

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=$PWD \
        python experiments/028-gene-essentiality/scripts/summary_figure.py
"""

from __future__ import annotations

import glob
import json
import os
import os.path as osp
from typing import Any

import matplotlib
import numpy as np
from dotenv import load_dotenv
from pydantic import BaseModel

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402

from torchcell.timestamp import timestamp  # noqa: E402
from torchcell.utils import (  # noqa: E402
    MAX_HEIGHT_MM,
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    mm_to_in,
    savefig_true_size_svg,
)

load_dotenv()
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
ASSET_IMAGES_DIR = os.environ["ASSET_IMAGES_DIR"]
RESULTS = osp.join(EXPERIMENT_ROOT, "028-gene-essentiality", "results")
IMAGES = osp.join(ASSET_IMAGES_DIR, "028-gene-essentiality")

#: The figure's one color code, held across every panel.
COMPARATOR = PLOT_PALETTE[5]
NO_SOLVE = PLOT_PALETTE[3]
SOLVE = PLOT_PALETTE[0]
GRAPH = PLOT_PALETTE[1]
AVERAGE = PLOT_PALETTE[2]

WIDTH_MM = PANEL_WIDTHS_MM["full"]


class Bar(BaseModel):
    """One horizontal bar: what it is, its value, and how it is drawn."""

    label: str
    value: float
    color: str
    interval: tuple[float, float] | None = None
    points: list[float] = []
    hatched: bool = False
    minutes: float | None = None


def read(name: str) -> Any:
    with open(osp.join(RESULTS, name)) as fh:
        return json.load(fh)


def place(fig: Figure, x: float, top: float, w: float, h: float) -> Axes:
    """An axes whose box is given in millimeters from the figure's top-left corner."""
    height_mm = fig.get_size_inches()[1] * 25.4
    ax = fig.add_axes(
        (x / WIDTH_MM, 1.0 - (top + h) / height_mm, w / WIDTH_MM, h / height_mm)
    )
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)
    ax.tick_params(width=0.5, length=2)
    return ax


def letter(fig: Figure, x: float, top: float, text: str) -> None:
    """A panel letter at the outer top-left of the panel's block, on the title line."""
    fig.text(
        x / WIDTH_MM,
        1.0 - (top - 1.0) / (fig.get_size_inches()[1] * 25.4),
        text,
        fontsize=8,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def auroc_axis(ax: Axes, low: float, high: float) -> None:
    """A gridline every 0.1 and a label every 0.2."""
    ax.set_xlim(low, high)
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.tick_params(axis="x", which="minor", length=0)
    ax.grid(axis="x", which="both", lw=0.3, color="0.85", zorder=0)
    ax.set_axisbelow(True)


def draw_bars(ax: Axes, bars: list[Bar], values: list[float] | None = None) -> None:
    """Bars top to bottom in the order given; seed points and intervals where present."""
    y = np.arange(len(bars))[::-1]
    shown = values if values is not None else [b.value for b in bars]
    ax.barh(
        y,
        shown,
        height=0.7,
        color=[b.color for b in bars],
        edgecolor="black",
        lw=0.4,
        hatch=None,
        zorder=3,
    )
    for yi, b, v in zip(y, bars, shown, strict=True):
        if b.hatched:
            ax.barh(
                yi,
                v,
                height=0.7,
                color="none",
                edgecolor="black",
                lw=0.4,
                hatch="/////",
                zorder=3.5,
            )
        if values is not None:
            continue
        if b.interval is not None:
            ax.errorbar(
                b.value,
                yi,
                xerr=[[b.value - b.interval[0]], [b.interval[1] - b.value]],
                fmt="none",
                ecolor="black",
                elinewidth=0.5,
                capsize=1.2,
                zorder=5,
            )
        if b.points:
            spread = (
                np.linspace(-0.22, 0.22, len(b.points)) if len(b.points) > 1 else [0.0]
            )
            ax.scatter(
                b.points, yi + np.array(spread), s=2.2, color="black", lw=0, zorder=6
            )
    ax.set_yticks(y)
    ax.set_yticklabels([b.label for b in bars])
    ax.set_ylim(-0.6, len(bars) - 0.4)
    ax.tick_params(axis="y", length=0)


# --------------------------------------------------------------------------- data


def fold_ensemble(name: str) -> tuple[float, tuple[float, float]]:
    e = read(name)
    return e["ensemble_auroc"], (e["ensemble_ci_low"], e["ensemble_ci_high"])


def leaderboard() -> list[Bar]:
    """Every method on the 195 released held-out genes, worst to best."""
    fcl = {
        s["method"]: s
        for s in read("fcl_comparison.json")["scored"]
        if s["gene_set"] == "ours_scored" and s["seed"] is None
    }
    ladder = max(read("logistic_baselines.json")["rows"], key=lambda r: r["val_auroc"])
    cheap = read("graph_baselines_folds.json")["arms"]
    go = read("graph_baselines_go_folds.json")["arms"]["GCN400"]
    families = read("family_ensembles_folds.json")["combinations"]

    def average(member: str) -> dict[str, Any]:
        return next(
            c
            for c in families
            if len(c["families"]) == 2
            and any(member in f for f in c["families"])
            and any("GO convolution" in f for f in c["families"])
        )

    bars = [
        Bar(
            label="FBA growth ratio, YPD",
            value=fcl["FBA"]["auroc"],
            color=COMPARATOR,
            interval=(fcl["FBA"]["ci_low"], fcl["FBA"]["ci_high"]),
        ),
        Bar(
            label="Flux Cone Learning, released",
            value=fcl["FCL"]["auroc"],
            color=COMPARATOR,
            interval=(fcl["FCL"]["ci_low"], fcl["FCL"]["ci_high"]),
        ),
        Bar(
            label=f"{ladder['model']} on GO + FBA ratios",
            value=ladder["test_auroc"],
            color=COMPARATOR,
            interval=(ladder["test_ci_low"], ladder["test_ci_high"]),
        ),
        Bar(
            label="MLP, three inputs",
            value=cheap["MLP"]["ensemble_auroc"],
            color=NO_SOLVE,
            interval=tuple(cheap["MLP"]["ensemble_ci"]),
        ),
        Bar(
            label="graph convolution, three inputs",
            value=cheap["GCN"]["ensemble_auroc"],
            color=GRAPH,
            interval=tuple(cheap["GCN"]["ensemble_ci"]),
        ),
    ]
    for name, label in [
        ("w23_chromprotcv_ensemble.json", "transformer + FBA ratios (reported)"),
        ("w41_pathfluxcv_ensemble.json", "transformer + learned-ceiling LP"),
        ("w43_pathoffcv_ensemble.json", "transformer + cached solve head"),
        ("w50_pathlogfreecv_ensemble.json", "transformer + layer-free solve"),
        ("w99_pathloggprcv_ensemble.json", "transformer + solve + rule counts"),
    ]:
        value, interval = fold_ensemble(name)
        bars.append(
            Bar(
                label=label,
                value=value,
                color=SOLVE,
                interval=interval,
                hatched="LP" in label,
            )
        )
    bars.append(
        Bar(
            label="graph convolution on GO",
            value=go["ensemble_auroc"],
            color=GRAPH,
            interval=tuple(go["ensemble_ci"]),
        )
    )
    for member, label in [
        ("cached solve", "GO graph + cached solve head"),
        ("layer-free log1p vector", "GO graph + layer-free solve"),
        ("GPR redundancy", "GO graph + solve + rule counts"),
    ]:
        c = average(member)
        bars.append(
            Bar(
                label=label,
                value=c["auroc"],
                color=AVERAGE,
                interval=(c["ci_low"], c["ci_high"]),
            )
        )
    return sorted(bars, key=lambda b: -b.value)


def cells(*patterns: str) -> tuple[list[float], float]:
    """Held-out AUROC per cell and the mean wall minutes, over every matching cell."""
    aurocs, minutes = [], []
    for pat in patterns:
        for p in sorted(glob.glob(osp.join(RESULTS, pat))):
            with open(p) as fh:
                d = json.load(fh)
            aurocs.append(float(d["test"]["pinned"]["auroc"]))
            minutes.append(float(d["wall_time_s"]) / 60.0)
    assert aurocs, patterns
    return aurocs, float(np.mean(minutes))


#: The module's forms on the pathway multi-hot with ProtT5 (the last on ProtT5 alone, the
#: only backbone the forced column ran on): label, result globs, color, LP in the loop.
MODULE_FORMS: list[tuple[str, tuple[str, ...], str, bool]] = [
    ("no metabolism", ("w39_pathpool_h12l1p1_e200_s*.json",), NO_SOLVE, False),
    (
        "FBA growth ratios on the logit",
        ("w38_protpathfba_h12l1p1_e200_s*.json",),
        SOLVE,
        False,
    ),
    (
        "solve vector, layer-free, on the logit",
        ("w49_pathfbapfbalogfree_h12l1p1_e200_s*.json",),
        SOLVE,
        False,
    ),
    (
        "+ gene-rule redundancy columns",
        ("w96_pathloggpr_h12l1p1_e200_s*.json",),
        SOLVE,
        False,
    ),
    (
        "solve vector on the node",
        ("w59_pathsolvenode_h12l1p1_e200_s*.json",),
        SOLVE,
        False,
    ),
    (
        "cached solve, flux head",
        ("w43_pathlpoffgateflux_h12l1p1_e200_s*.json",),
        SOLVE,
        True,
    ),
    (
        "learned ceilings, flux head",
        ("w39_pathlpelgateflux_h12l1p1_e200_s*.json",),
        SOLVE,
        True,
    ),
    (
        "learned ceilings, gated growth",
        ("w39_pathlpelgate100_h12l1p1_e200_s*.json",),
        SOLVE,
        True,
    ),
    (
        "learned ceilings, forced growth",
        (
            "w30_lpcapbio100_h12l1p1_e200_s*.json",
            "w88_lpcapbio100_h12l1p1_e200_s*.json",
            "w89_lpcapbio100_h12l1p1_e200_s*.json",
        ),
        SOLVE,
        True,
    ),
]


def module_forms() -> list[Bar]:
    bars = []
    for label, patterns, color, in_loop in MODULE_FORMS:
        aurocs, minutes = cells(*patterns)
        bars.append(
            Bar(
                label=label,
                value=float(np.mean(aurocs)),
                color=color,
                points=aurocs,
                hatched=in_loop,
                minutes=minutes,
            )
        )
    return bars


def genome_transfer() -> list[Bar]:
    arms = {a["label"]: a for a in read("genome_heldout_readout.json")["arms"]}
    keep = [
        (
            "graph convolution, physical graph, degree and clustering",
            "graph convolution, degree only",
            GRAPH,
        ),
        (
            "graph convolution, union of nine, ProtT5",
            "graph convolution, ProtT5",
            GRAPH,
        ),
        (
            "graph convolution on the three inputs, 400 epochs",
            "graph convolution, three inputs",
            GRAPH,
        ),
        ("graph convolution on GO alone, 400 epochs", "graph convolution on GO", GRAPH),
        (
            "pathway multi-hot + ProtT5 + FBA affine (round 22 recipe)",
            "transformer + FBA ratios",
            SOLVE,
        ),
        (
            "layer-free log1p vector + affine, pathway + ProtT5 (round 33 form)",
            "transformer + layer-free solve",
            SOLVE,
        ),
        (
            "cached solve + flux head, pathway + ProtT5 (round 27 form)",
            "transformer + cached solve head",
            SOLVE,
        ),
        ("MLP on the three inputs", "MLP, three inputs", NO_SOLVE),
    ]
    bars = [
        Bar(
            label=label,
            value=arms[k]["pooled_auroc"],
            color=color,
            interval=tuple(arms[k]["pooled_ci"]),
        )
        for k, label, color in keep
    ]
    raw = next(
        r
        for r in read("degree_by_class.json")["reads"]
        if r["graph"] == "physical" and r["n_genes"] == 4674
    )
    bars.append(
        Bar(
            label="physical degree, nothing trained",
            value=raw["auroc"],
            color=COMPARATOR,
            interval=tuple(raw["ci"]),
        )
    )
    return sorted(bars, key=lambda b: -b.value)


def store_scale() -> list[Bar]:
    k = read("store_key_count.json")
    b = read("solid_growth_store_SC_ipm.json")
    return [
        Bar(label="records in the build", value=k["n_records"], color=COMPARATOR),
        Bar(
            label="records reading the wild-type row",
            value=k["n_records_empty_gem_subset"],
            color=COMPARATOR,
        ),
        Bar(
            label="programs solved, once each", value=k["n_distinct_keys"], color=SOLVE
        ),
        Bar(label="of them lethal under the solve", value=b["n_lethal"], color=SOLVE),
        Bar(
            label="of them uncertified, stored zero",
            value=b["n_uncertified_zero_flux"],
            color=SOLVE,
        ),
    ]


#: Per held-out set: the two models read, as (label in the report, color).
ENRICH_SETS: list[tuple[str, str, list[tuple[str, str]]]] = [
    (
        "released 195 (metabolic genes)",
        "Misranked genes, released 195",
        [("transformer, layer-free solve", SOLVE), ("graph convolution on GO", GRAPH)],
    ),
    (
        "genome-wide 4,674 (no GEM reaction)",
        "Misranked genes, genome-wide 4,674",
        [
            ("transformer, layer-free solve", SOLVE),
            ("graph convolution on ProtT5", GRAPH),
        ],
    ),
]
CLASSES = ["essential, ranked low", "nonessential, ranked high"]
#: Terms drawn per class: the named features that were tested, then each model's best.
TERMS_PER_CLASS = 4


def enrichment_rows(
    set_label: str, models: list[tuple[str, str]]
) -> list[dict[str, Any]]:
    """Per class, the terms to draw and each model's (effect, q) on them."""
    fams = {
        (f["model"], f["gene_class"]): f
        for f in read("misclass_enrichment.json")["families"]
        if f["held_out_set"] == set_label
    }
    rows: list[dict[str, Any]] = []
    for cls in CLASSES:
        chosen: list[tuple[str, str]] = []
        for model, _ in models:
            for r in fams[(model, cls)]["named_features"]:
                if (r["term"], r["name"]) not in chosen:
                    chosen.append((r["term"], r["name"]))
        rank = 0
        while len(chosen) < TERMS_PER_CLASS + 1 and rank < 12:
            for model, _ in models:
                top = fams[(model, cls)]["top_specific"]
                if (
                    rank < len(top)
                    and (top[rank]["term"], top[rank]["name"]) not in chosen
                ):
                    chosen.append((top[rank]["term"], top[rank]["name"]))
            rank += 1
        for term, name in chosen[: TERMS_PER_CLASS + 1]:
            rows.append(
                {
                    "class": cls,
                    "name": name,
                    "by_model": {
                        m: fams[(m, cls)]["all_terms"].get(term) for m, _ in models
                    },
                }
            )
    return rows


def panel_enrichment(ax: Axes, set_label: str, models: list[tuple[str, str]]) -> None:
    rows = enrichment_rows(set_label, models)
    y = np.arange(len(rows))[::-1].astype(float)
    # A gap between the two classes, so the group reads without a rule inside the axes.
    first_second = next(i for i, r in enumerate(rows) if r["class"] == CLASSES[1])
    y[:first_second] += 0.8
    for yi, row in zip(y, rows, strict=True):
        for k, (model, color) in enumerate(models):
            got = row["by_model"][model]
            if got is None:
                continue
            effect, q = got
            ax.scatter(
                effect,
                yi + (0.16 if k == 0 else -0.16),
                s=9,
                marker="o" if k == 0 else "s",
                facecolor=color if q < 0.1 else "white",
                edgecolor=color,
                lw=0.7,
                zorder=4,
            )
    ax.axvline(0.5, color="black", lw=0.5, ls=(0, (2, 2)), zorder=2)
    ax.set_yticks(y)
    # Full term names: a truncated GO term cannot be looked up.
    ax.set_yticklabels([r["name"] for r in rows])
    ax.set_ylim(-0.7, y.max() + 0.7)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0.2, 1.0)
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.tick_params(axis="x", which="minor", length=0)
    ax.grid(axis="x", which="both", lw=0.3, color="0.85", zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlabel("P(annotated gene misranked worse)")
    # The class names sit outside the axes, right of each group.
    for cls, members in (
        (CLASSES[0], y[:first_second]),
        (CLASSES[1], y[first_second:]),
    ):
        ax.annotate(
            cls.replace(", ", ",\n"),
            xy=(1.03, float(np.mean(members))),
            xycoords=("axes fraction", "data"),
            va="center",
            ha="left",
            fontsize=6,
        )


# --------------------------------------------------------------------------- figure


def new_figure(height_mm: float) -> Figure:
    assert height_mm <= MAX_HEIGHT_MM
    return plt.figure(figsize=(mm_to_in(WIDTH_MM), mm_to_in(height_mm)))


def save(fig: Figure, name: str) -> None:
    os.makedirs(IMAGES, exist_ok=True)
    stem = osp.join(IMAGES, f"{name}_{timestamp()}")
    savefig_true_size_svg(fig, stem + ".svg")
    fig.savefig(stem + ".png", dpi=300)
    print(stem + ".svg")


def patch(color: str, label: str, hatch: str | None = None) -> Patch:
    return Patch(facecolor=color, edgecolor="black", lw=0.4, hatch=hatch, label=label)


def top_legend(fig: Figure, handles: list[Any], ncol: int) -> None:
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.0, 1.0),
        ncol=ncol,
        frameon=False,
        handlelength=1.2,
        handletextpad=0.4,
        columnspacing=1.0,
        borderaxespad=0.2,
    )


def goal_one() -> None:
    """Goal 1, panels b to e; panel a is the draw.io explainer placed above in LaTeX."""
    height = 120.0
    fig = new_figure(height)
    top_legend(
        fig,
        [
            patch(COMPARATOR, "comparator, nothing learned here"),
            patch(NO_SOLVE, "learned, no solve, no graph"),
            patch(SOLVE, "family 1: transformer reading the solve"),
            patch(GRAPH, "family 2: message passing on gene graphs"),
            patch(AVERAGE, "equal-weight average of families 1 and 2"),
            patch("white", "LP solved inside training", hatch="/////"),
        ],
        ncol=3,
    )
    left_label, right_block = 47.0, 92.0

    top = 15.0
    ax = place(fig, left_label, top, 40.0, 46.0)
    draw_bars(ax, leaderboard())
    auroc_axis(ax, 0.5, 1.0)
    ax.set_xlabel("AUROC, 195 held-out genes, fold ensembles")
    ax.set_title("Best method, released 195 genes", pad=3, loc="left")
    letter(fig, 0.0, top, "b")

    # c: the finalists on the ten repeated outer splits, fixed in advance (round 75).
    resplit = read("resplit_readout.json")["splits"]
    order = sorted(range(len(resplit)), key=lambda i: resplit[i]["average"])
    ax = place(fig, right_block + 12.0, top, 72.0, 46.0)
    for key, color, marker in (
        ("layer_free", SOLVE, "o"),
        ("go_convolution", GRAPH, "s"),
        ("average", AVERAGE, "D"),
    ):
        ax.plot(
            range(len(order)),
            [resplit[i][key] for i in order],
            ls="-",
            lw=0.5,
            marker=marker,
            ms=3.0,
            color=color,
            mec="black",
            mew=0.4,
            zorder=3,
        )
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([str(resplit[i]["resplit"]) for i in order])
    ax.set_ylim(0.7, 1.0)
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))
    ax.tick_params(axis="y", which="minor", length=0)
    ax.grid(axis="y", which="both", lw=0.3, color="0.85", zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlabel("repeated outer split, sorted by the average's AUROC")
    ax.set_ylabel("held-out AUROC")
    ax.set_title(
        "Ten other held-out fifths, models fixed in advance", pad=3, loc="left"
    )
    letter(fig, right_block, top, "c")

    top = 78.0
    ax = place(fig, left_label, top, 40.0, 30.0)
    draw_bars(ax, genome_transfer())
    ax.axvline(0.5, color="black", lw=0.5, ls=(0, (2, 2)), zorder=5)
    auroc_axis(ax, 0.4, 0.8)
    ax.set_xlabel("AUROC, 4,674 genome-wide held-out genes")
    ax.set_title("What transfers off the metabolic genes", pad=3, loc="left")
    letter(fig, 0.0, top, "d")

    set_label, title, models = ENRICH_SETS[0]
    markers = [
        Line2D(
            [],
            [],
            marker="o",
            ls="",
            ms=3.2,
            mfc=SOLVE,
            mec=SOLVE,
            mew=0.7,
            label="family 1",
        ),
        Line2D(
            [],
            [],
            marker="s",
            ls="",
            ms=3.2,
            mfc=GRAPH,
            mec=GRAPH,
            mew=0.7,
            label="family 2",
        ),
        Line2D(
            [],
            [],
            marker="o",
            ls="",
            ms=3.2,
            mfc="white",
            mec="black",
            mew=0.7,
            label="hollow: q at or above 0.1",
        ),
    ]
    fig.legend(
        handles=markers,
        loc="upper left",
        bbox_to_anchor=(right_block / WIDTH_MM, 1.0 - (top - 9.0) / height),
        ncol=3,
        frameon=False,
        handletextpad=0.3,
        columnspacing=1.2,
        borderaxespad=0.2,
    )
    ax = place(fig, right_block + 50.0, top, 20.0, 30.0)
    panel_enrichment(ax, set_label, models)
    ax.set_title(title, pad=3, loc="left")
    letter(fig, right_block, top, "e")
    save(fig, "summary_goal1")


#: Row label -> (ladder key, repeated-split pair against no metabolism), for goal 2.
PORTABLE_FORMS: list[tuple[str, str, str | None]] = [
    (
        "1 medium: ratio, flux vector, rule counts",
        "one_solve_counts",
        "one medium, flux vector and counts, over no metabolism",
    ),
    (
        "4 media: ratios, flux vector, rule counts",
        "four_solve_counts",
        "four media, flux vector and counts, over no metabolism",
    ),
    (
        "4 media: ratios, flux vector",
        "four_solve",
        "four media and the flux vector, over no metabolism",
    ),
    (
        "4 media: ratios, rule counts",
        "four_counts",
        "four media and the counts, over no metabolism",
    ),
    (
        "2 media: ratios, rule counts",
        "two_counts",
        "two media and the counts, over no metabolism",
    ),
    (
        "1 medium: ratio, rule counts",
        "one_counts",
        "one medium and the counts, over no metabolism",
    ),
    ("4 media: ratios alone", "four_only", "four ratios alone, over no metabolism"),
    ("no metabolism", "none", None),
]


def goal_two() -> None:
    """Goal 2: which module form helps, what it costs, and the portable forms."""
    height = 124.0
    fig = new_figure(height)
    top_legend(
        fig,
        [
            patch(NO_SOLVE, "learned, no metabolic column"),
            patch(SOLVE, "transformer reading an FBA-derived column"),
            patch(COMPARATOR, "count of records or programs"),
            patch("white", "LP solved inside training", hatch="/////"),
            Line2D(
                [],
                [],
                marker="o",
                ls="",
                ms=2.2,
                mfc="white",
                mec="black",
                mew=0.4,
                label="one seed (a, d) or one split (e)",
            ),
            Line2D(
                [],
                [],
                marker="D",
                ls="",
                ms=2.8,
                mfc="black",
                mec="black",
                label="five-fold ensemble",
            ),
        ],
        ncol=3,
    )
    left_label, right_block = 47.0, 96.0

    top = 15.0
    forms = module_forms()
    ax = place(fig, left_label, top, 26.0, 46.0)
    draw_bars(ax, forms)
    auroc_axis(ax, 0.4, 1.0)
    ax.set_xlabel("AUROC per cell, seed mean")
    ax.set_title("Which module form", pad=3, loc="left")
    letter(fig, 0.0, top, "a")
    axb = place(fig, left_label + 26.0 + 7.0, top, 12.0, 46.0)
    draw_bars(axb, forms, values=[b.minutes or 0.0 for b in forms])
    axb.set_yticklabels([])
    axb.set_xlim(0, 160)
    axb.xaxis.set_major_locator(MultipleLocator(60))
    axb.grid(axis="x", lw=0.3, color="0.85", zorder=0)
    axb.set_axisbelow(True)
    axb.set_xlabel("minutes")
    axb.set_title("Cost per cell", pad=3, loc="left")
    letter(fig, left_label + 26.0 + 3.0, top, "b")

    ax = place(fig, right_block + 40.0, top, 38.0, 24.0)
    draw_bars(ax, store_scale())
    ax.set_xscale("log")
    ax.set_xlim(1e3, 5e7)
    ax.grid(axis="x", lw=0.3, color="0.85", zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlabel("count, 025 solid-growth build")
    ax.set_title("The layer-free solve at database scale", pad=3, loc="left")
    letter(fig, right_block, top, "c")

    # Row 2: the forms a record of one genotype in one medium can carry.
    report = read("simplification_readout.json")
    ladder = {r["key"]: r for r in report["ladder"]}
    pairs = {r["name"]: r for r in report["resplit"]}
    y = np.arange(len(PORTABLE_FORMS))[::-1]
    colors = [NO_SOLVE if k == "none" else SOLVE for _, k, _ in PORTABLE_FORMS]
    top, h = 80.0, 36.0

    ax = place(fig, 50.0, top, 40.0, h)
    for yi, color, (_, key, _) in zip(y, colors, PORTABLE_FORMS, strict=True):
        row = ladder[key]
        ax.barh(
            yi,
            row["cell_mean"],
            height=0.62,
            color=color,
            edgecolor="black",
            lw=0.4,
            zorder=2,
        )
        ax.plot(
            row["cell_values"],
            np.full(len(row["cell_values"]), yi),
            "o",
            ms=2.2,
            mfc="white",
            mec="black",
            mew=0.4,
            zorder=3,
        )
        fold = row["fold_ensemble_three_seeds"] or row["fold_ensemble_one_seed"]
        if fold is not None:
            ax.plot(fold, yi, "D", ms=2.8, color="black", zorder=4)
    ax.set_xlim(0.8, 0.96)
    ax.xaxis.set_major_locator(MultipleLocator(0.05))
    ax.grid(axis="x", lw=0.3, color="0.85", zorder=0)
    ax.set_axisbelow(True)
    ax.set_yticks(y)
    ax.set_yticklabels([label for label, _, _ in PORTABLE_FORMS])
    ax.set_ylim(-0.6, len(PORTABLE_FORMS) - 0.4)
    ax.set_xlabel("held-out AUROC, released split")
    ax.set_title("Released 195 genes", pad=3, loc="left")
    letter(fig, 0.0, top, "d")

    ax = place(fig, 98.0, top, 42.0, h)
    for yi, color, (_, _, name) in zip(y, colors, PORTABLE_FORMS, strict=True):
        if name is None or name not in pairs:
            continue
        row = pairs[name]
        ax.barh(
            yi,
            row["mean_difference"],
            height=0.62,
            color=color,
            edgecolor="black",
            lw=0.4,
            zorder=2,
        )
        ax.plot(
            row["per_unit_difference"],
            np.full(row["n"], yi),
            "o",
            ms=2.2,
            mfc="white",
            mec="black",
            mew=0.4,
            zorder=3,
        )
    ax.axvline(0.0, color="black", lw=0.5, zorder=1)
    ax.set_xlim(-0.03, 0.13)
    ax.xaxis.set_major_locator(MultipleLocator(0.05))
    ax.grid(axis="x", lw=0.3, color="0.85", zorder=0)
    ax.set_axisbelow(True)
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.set_ylim(-0.6, len(PORTABLE_FORMS) - 0.4)
    ax.set_xlabel("AUROC minus no metabolism, ten repeated splits")
    ax.set_title("Gain over the same cell with no metabolism", pad=3, loc="left")
    letter(fig, 94.0, top, "e")

    ax = place(fig, 148.0, top, 27.0, h)
    for yi, color, (_, _, name) in zip(y, colors, PORTABLE_FORMS, strict=True):
        if name is None or name not in pairs:
            continue
        row = pairs[name]
        ax.barh(
            yi,
            row["n_pairs_positive"] / row["n_pairs"],
            height=0.62,
            color=color,
            edgecolor="black",
            lw=0.4,
            zorder=2,
        )
    ax.axvline(0.5, color="black", lw=0.5, ls=(0, (2, 2)), zorder=1)
    auroc_axis(ax, 0.0, 1.0)
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.set_ylim(-0.6, len(PORTABLE_FORMS) - 0.4)
    ax.set_xlabel("share of split-and-seed pairs won")
    ax.set_title("How often the module helps", pad=3, loc="left")
    letter(fig, 144.0, top, "f")
    save(fig, "summary_goal2")


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 6,
            "svg.fonttype": "none",
            "axes.linewidth": 0.5,
            "hatch.linewidth": 0.4,
            "axes.titlesize": 6,
        }
    )
    goal_one()
    goal_two()


if __name__ == "__main__":
    main()
