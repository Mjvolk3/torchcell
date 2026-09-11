# experiments/026-metabolism-flux/scripts/plot_external_pka.py
# [[experiments.026-metabolism-flux.scripts.plot_external_pka]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/026-metabolism-flux/scripts/plot_external_pka.py

r"""Six panels of the open-source pKa benchmark, drawn from the per-compound scores.

``external_pka_benchmark.py score`` writes one row per compound per tool to
``results/external_pka_per_compound.csv``: the cache's ChemAxon pKa list, the predicted
list, and the change in ``dGf'0`` at pH 7.2 that swapping one for the other produces. The
tables in the note report the means of that file. The panels here report the file itself,
so the 300-compound claim is shown rather than stated.

Panels

``a``
    Predicted against cache pKa, paired by rank over the shorter list, one panel per tool,
    with the identity line. Rank pairing is the same rule the scoring uses.
``b``
    The signed change in ``dGf'0`` per compound, by tool. Signed rather than absolute,
    because the direction is the interesting part: a predictor that misses low on every
    compound is a different problem from one that scatters.
``c``
    The same signed change split by how many pKas the cache holds, which is where
    pKasolver-lite and MolGpKa separate.
``d``
    The same signed change split by functional group.
``e``
    The fraction of compounds whose predicted pKa count inside (0, 14) equals the cache's.
``f``
    Per-compound signed shifts of one tool against another, so the predictors can be
    compared to each other and not only to ChemAxon. A point on the identity line is a
    compound the two tools misprice identically; a point on an axis is one that only one
    of them misprices.

Every number a panel shows is written to ``results/external_pka_figure.json``, and the
per-compound signed shifts are pivoted one column per tool into
``results/external_pka_shift_by_compound.csv``.

Run::

    PYTHONPATH=<repo> ~/miniconda3/envs/torchcell/bin/python \
        experiments/026-metabolism-flux/scripts/plot_external_pka.py
"""

from __future__ import annotations

import argparse
import os
import os.path as osp
from datetime import UTC, datetime
from typing import cast

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator
from pydantic import BaseModel

from torchcell.utils import (
    MAX_HEIGHT_MM,
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    apply_paper_style,
    mm_to_in,
    savefig_true_size_svg,
)

load_dotenv()

EXPERIMENT_ROOT = cast(str, os.environ["EXPERIMENT_ROOT"])
ASSET_IMAGES_DIR = cast(str, os.environ["ASSET_IMAGES_DIR"])
RESULTS = osp.join(EXPERIMENT_ROOT, "026-metabolism-flux", "results")
PER_COMPOUND_CSV = osp.join(RESULTS, "external_pka_per_compound.csv")
FIGURE_JSON = osp.join(RESULTS, "external_pka_figure.json")
SHIFT_CSV = osp.join(RESULTS, "external_pka_shift_by_compound.csv")
IMAGE_DIR = osp.join(ASSET_IMAGES_DIR, "026-metabolism-flux")
FIGURE_STEM = osp.join(IMAGE_DIR, "external_pka_benchmark")

# The figure is laid out in millimeters on a full-width Nature canvas, so every panel
# lands on a standard width and the whole page can be dropped into draw.io at true size.
FIG_W_MM = PANEL_WIDTHS_MM["full"]
# The bottom row carries rotated category labels under an axis label, so the canvas keeps
# 23 mm below the last panel; a shorter canvas silently clips them.
FIG_H_MM = 168.0

MIN_PH = 0.0
MAX_PH = 14.0
P_H = 7.2

STRATUM_ORDER = [f"n_pka_{k}" for k in range(1, 7)]
GROUP_ORDER = ["carboxylate", "amine", "phenol", "phosphate"]
# The tool key is an identifier: it names a prediction file, a CSV column and a JSON key,
# and it stays lowercase and hyphenated everywhere those are read. It is not a label. Every
# place a reader sees a tool named -- panel titles, the legend, tick labels, axis labels --
# uses the published capitalization instead, and the two are kept apart on purpose so a
# rename of one never silently changes the other.
TOOL_DISPLAY_NAMES = {
    "pkasolver-lite": "pKasolver-lite",
    "molgpka": "MolGpKa",
    "unipka": "Uni-pKa full template",
    "unipka-simple-template": "Uni-pKa simple template",
    "qupkake": "QupKake",
    "qupkake-truncated": "QupKake truncated",
    "chemaxon-cache": "ChemAxon cache",
}


def display_tool(key: str) -> str:
    """The published name of a tool key, or the key when none is registered."""
    return TOOL_DISPLAY_NAMES.get(key, key)


# A tool with a handful of scored compounds cannot carry a box plot or a scatter cloud,
# and putting one in the figure would show a spread that is an artifact of its n. The
# excluded tools are named in the JSON rather than dropped silently.
MIN_ROWS_FOR_PANEL = 30


class PanelNumbers(BaseModel):
    """Everything one panel shows, so the caption never has to be read off the pixels."""

    panel: str
    what: str
    values: dict[str, object]


class FigureNumbers(BaseModel):
    """The figure as data."""

    built_at: str
    source_csv: str
    figure_svg: str
    figure_png: str
    conditions: dict[str, float]
    tools_drawn: list[str]
    tools_excluded: dict[str, int]
    panels: list[PanelNumbers]


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


def _parse_list(text: object) -> list[float]:
    if not isinstance(text, str) or not text:
        return []
    return [float(v) for v in text.split(";")]


def _mm_axes(fig: plt.Figure, x: float, y: float, w: float, h: float) -> plt.Axes:
    """An axes placed in millimeters from the top-left of the canvas."""
    return fig.add_axes(
        [x / FIG_W_MM, 1.0 - (y + h) / FIG_H_MM, w / FIG_W_MM, h / FIG_H_MM]
    )


def _panel_letter(ax: plt.Axes, letter: str, dx_mm: float = -9.0) -> None:
    """The panel letter at the axes' OUTER top-left, clear of every drawn element."""
    ax.annotate(
        letter,
        xy=(0, 1),
        xycoords="axes fraction",
        xytext=(dx_mm / 25.4 * 72, 8),
        textcoords="offset points",
        fontsize=8,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def _box(
    ax: plt.Axes,
    positions: list[float],
    data: list[np.ndarray],
    colors: list[str],
    width: float,
) -> None:
    """Boxes with black structure and a solid palette face, fliers kept as small dots."""
    artists = ax.boxplot(
        data,
        positions=positions,
        widths=width,
        patch_artist=True,
        showfliers=True,
        whis=(5, 95),
        flierprops={
            "marker": ".",
            "markersize": 1.0,
            "markerfacecolor": "black",
            "markeredgecolor": "none",
            "alpha": 0.5,
        },
        medianprops={"color": "black", "linewidth": 0.6},
        boxprops={"linewidth": 0.4, "edgecolor": "black"},
        whiskerprops={"linewidth": 0.4, "color": "black"},
        capprops={"linewidth": 0.4, "color": "black"},
    )
    for patch, color in zip(artists["boxes"], colors):
        patch.set_facecolor(color)


def _symlog(ax: plt.Axes) -> None:
    """A symmetric log energy axis, so a 30 kJ/mol tail and a 0.1 kJ/mol median coexist."""
    ax.set_yscale("symlog", linthresh=1.0, linscale=0.6)
    ax.set_yticks([-10, -1, 0, 1, 10])
    ax.set_yticklabels(["-10", "-1", "0", "1", "10"])
    ax.axhline(0.0, color="#666666", linewidth=0.3, zorder=0)


def _stats(values: np.ndarray) -> dict[str, object]:
    if values.size == 0:
        return {"n": 0}
    return {
        "n": int(values.size),
        "median": float(np.median(values)),
        "q1": float(np.percentile(values, 25)),
        "q3": float(np.percentile(values, 75)),
        "p5": float(np.percentile(values, 5)),
        "p95": float(np.percentile(values, 95)),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean_abs": float(np.mean(np.abs(values))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=PER_COMPOUND_CSV)
    parser.add_argument(
        "--tools",
        nargs="+",
        default=None,
        help="draw only these tools, in this order; default is every tool with enough rows",
    )
    parser.add_argument(
        "--reference-tool",
        default="pkasolver-lite",
        help="the x axis of panel f; every other drawn tool is plotted against it",
    )
    args = parser.parse_args()

    apply_paper_style()
    df = pd.read_csv(args.csv)
    df["cache_list"] = df["cache_pkas"].map(_parse_list)
    df["pred_list"] = df["predicted_pkas"].map(_parse_list)

    counts = df[df["ok"]].groupby("tool").size().to_dict()
    tools = [t for t in df["tool"].unique() if counts.get(t, 0) >= MIN_ROWS_FOR_PANEL]
    excluded = {
        t: int(counts.get(t, 0))
        for t in df["tool"].unique()
        if counts.get(t, 0) < MIN_ROWS_FOR_PANEL
    }
    if args.tools:
        missing = [t for t in args.tools if t not in set(df["tool"])]
        if missing:
            raise SystemExit(
                f"no rows for {missing}; present: {sorted(set(df['tool']))}"
            )
        excluded.update(
            {t: int(counts.get(t, 0)) for t in tools if t not in args.tools}
        )
        tools = list(args.tools)
    color_of = {tool: PLOT_PALETTE[i] for i, tool in enumerate(tools)}
    panels: list[PanelNumbers] = []

    fig = plt.figure(figsize=(mm_to_in(FIG_W_MM), mm_to_in(FIG_H_MM)))

    # -------------------------------------------------------------- a, paired pKas
    row_a_y, row_a_h = 10.0, 34.0
    left, right = 11.0, 3.0
    # The gap has to shrink as tools are added or the scatter panels fall under the width
    # where a 0 to 14 axis with 6 pt labels is legible. Five tools at the three-panel gap
    # would leave 22.6 mm each.
    gap = {1: 13.0, 2: 13.0, 3: 13.0, 4: 11.0}.get(len(tools), 8.0)
    w_a = (FIG_W_MM - left - right - gap * (len(tools) - 1)) / len(tools)
    panel_a: dict[str, object] = {}
    for i, tool in enumerate(tools):
        ax = _mm_axes(fig, left + i * (w_a + gap), row_a_y, w_a, row_a_h)
        sub = df[(df["tool"] == tool) & df["ok"]]
        xs: list[float] = []
        ys: list[float] = []
        for cache, pred in zip(sub["cache_list"], sub["pred_list"]):
            for k in range(min(len(cache), len(pred))):
                xs.append(cache[k])
                ys.append(pred[k])
        ax.plot([MIN_PH, MAX_PH], [MIN_PH, MAX_PH], color="#666666", linewidth=0.4)
        ax.scatter(
            xs, ys, s=1.2, color=color_of[tool], alpha=0.45, linewidths=0, zorder=3
        )
        ax.set_xlim(MIN_PH, MAX_PH)
        ax.set_ylim(MIN_PH, MAX_PH)
        ax.set_xticks([0, 4, 8, 12])
        ax.set_yticks([0, 4, 8, 12])
        ax.set_xlabel("cache pKa")
        if i == 0:
            ax.set_ylabel("predicted pKa")
            _panel_letter(ax, "a")
        ax.set_title(display_tool(tool), pad=2.0)
        arr_x, arr_y = np.asarray(xs), np.asarray(ys)
        panel_a[tool] = {
            "n_paired_pkas": int(arr_x.size),
            "n_compounds": int(sub.shape[0]),
            "mae": float(np.mean(np.abs(arr_x - arr_y))) if arr_x.size else None,
            "pearson_r": (
                float(np.corrcoef(arr_x, arr_y)[0, 1]) if arr_x.size > 1 else None
            ),
            "frac_within_1_unit": (
                float(np.mean(np.abs(arr_x - arr_y) <= 1.0)) if arr_x.size else None
            ),
        }
    panels.append(
        PanelNumbers(
            panel="a",
            what="predicted against cache pKa, paired by rank over the shorter list",
            values=panel_a,
        )
    )

    # ------------------------------------------------------------------- b, by tool
    row_b_y, row_b_h = 61.0, 32.0
    ax_b = _mm_axes(fig, left, row_b_y, 40.0, row_b_h)
    data_b = [
        df[(df["tool"] == t) & df["ok"]]["ddgf_prime_kj_per_mol"].dropna().to_numpy()
        for t in tools
    ]
    _box(ax_b, list(range(len(tools))), data_b, [color_of[t] for t in tools], 0.6)
    _symlog(ax_b)
    ax_b.set_xticks(range(len(tools)))
    ax_b.set_xticklabels([display_tool(t) for t in tools], rotation=30, ha="right")
    ax_b.set_ylabel(f"\u0394 dGf'0 at pH {P_H:g} (kJ/mol)")
    _panel_letter(ax_b, "b")
    panels.append(
        PanelNumbers(
            panel="b",
            what=f"signed change in dGf'0 at pH {P_H:g} per compound, by tool",
            values={t: _stats(d) for t, d in zip(tools, data_b)},
        )
    )

    # ---------------------------------------------------------------- c, by stratum
    ax_c = _mm_axes(fig, left + 55.0, row_b_y, FIG_W_MM - left - 55.0 - right, row_b_h)
    strata = [s for s in STRATUM_ORDER if (df["stratum"] == s).any()]
    positions: list[float] = []
    data_c: list[np.ndarray] = []
    colors_c: list[str] = []
    span = 0.8
    step = span / len(tools)
    panel_c: dict[str, object] = {}
    for j, stratum in enumerate(strata):
        for i, tool in enumerate(tools):
            sub = df[(df["tool"] == tool) & df["ok"] & (df["stratum"] == stratum)]
            values = sub["ddgf_prime_kj_per_mol"].dropna().to_numpy()
            positions.append(j + (i - (len(tools) - 1) / 2) * step)
            data_c.append(values)
            colors_c.append(color_of[tool])
            panel_c[f"{stratum}|{tool}"] = _stats(values)
    _box(ax_c, positions, data_c, colors_c, step * 0.82)
    _symlog(ax_c)
    ax_c.set_xlim(-0.6, len(strata) - 0.4)
    ax_c.set_xticks(range(len(strata)))
    ax_c.set_xticklabels([s.replace("n_pka_", "") for s in strata])
    ax_c.set_xlabel("number of cache pKas")
    ax_c.set_ylabel("\u0394 dGf'0 (kJ/mol)")
    _panel_letter(ax_c, "c")
    panels.append(
        PanelNumbers(
            panel="c",
            what="signed change in dGf'0 by number of cache pKas, per tool",
            values=panel_c,
        )
    )

    # ------------------------------------------------------------------ d, by group
    row_c_y, row_c_h = 113.0, 32.0
    ax_d = _mm_axes(fig, left, row_c_y, 66.0, row_c_h)
    groups = [
        g
        for g in GROUP_ORDER
        if df["groups"].fillna("").str.split("|").map(lambda v: g in v).any()
    ]
    positions = []
    data_d: list[np.ndarray] = []
    colors_d: list[str] = []
    panel_d: dict[str, object] = {}
    for j, group in enumerate(groups):
        has_group = df["groups"].fillna("").str.split("|").map(lambda v: group in v)
        for i, tool in enumerate(tools):
            sub = df[(df["tool"] == tool) & df["ok"] & has_group]
            values = sub["ddgf_prime_kj_per_mol"].dropna().to_numpy()
            positions.append(j + (i - (len(tools) - 1) / 2) * step)
            data_d.append(values)
            colors_d.append(color_of[tool])
            panel_d[f"{group}|{tool}"] = _stats(values)
    _box(ax_d, positions, data_d, colors_d, step * 0.82)
    _symlog(ax_d)
    ax_d.set_xlim(-0.6, len(groups) - 0.4)
    ax_d.set_xticks(range(len(groups)))
    ax_d.set_xticklabels(groups, rotation=20, ha="right")
    ax_d.set_ylabel("\u0394 dGf'0 (kJ/mol)")
    _panel_letter(ax_d, "d")
    panels.append(
        PanelNumbers(
            panel="d",
            what="signed change in dGf'0 by functional group, per tool",
            values=panel_d,
        )
    )

    # ------------------------------------------------------------- e, right pKa count
    ax_e = _mm_axes(fig, left + 78.0, row_c_y, 30.0, row_c_h)
    fractions = []
    panel_e: dict[str, object] = {}
    for tool in tools:
        sub = df[(df["tool"] == tool) & df["ok"]]
        frac = float(sub["count_matches"].mean())
        fractions.append(frac)
        panel_e[tool] = {
            "n": int(sub.shape[0]),
            "n_count_match": int(sub["count_matches"].sum()),
            "frac_count_match": frac,
            "mean_pred_minus_cache_count": float(
                (sub["n_pred_pkas"] - sub["n_cache_pkas"]).mean()
            ),
        }
    ax_e.bar(
        range(len(tools)),
        fractions,
        color=[color_of[t] for t in tools],
        edgecolor="black",
        linewidth=0.4,
        width=0.7,
    )
    ax_e.set_ylim(0, 1)
    ax_e.yaxis.set_major_locator(MultipleLocator(0.2))
    ax_e.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax_e.tick_params(axis="y", which="minor", length=0)
    ax_e.grid(axis="y", which="both", color="#CACACA", linewidth=0.3, zorder=0)
    ax_e.set_axisbelow(True)
    ax_e.set_xticks(range(len(tools)))
    ax_e.set_xticklabels([display_tool(t) for t in tools], rotation=30, ha="right")
    ax_e.set_ylabel("fraction with the cache's pKa count")
    _panel_letter(ax_e, "e")
    panels.append(
        PanelNumbers(
            panel="e",
            what="fraction of compounds whose pKa count in (0, 14) equals the cache's",
            values=panel_e,
        )
    )

    # --------------------------------------------------------------- f, tool vs tool
    ax_f = _mm_axes(fig, left + 128.0, row_c_y, 40.0, row_c_h)
    reference = args.reference_tool
    wide = df[df["ok"]].pivot_table(
        index="compound_id", columns="tool", values="ddgf_prime_kj_per_mol"
    )
    others = [t for t in tools if t != reference]
    panel_f: dict[str, object] = {}
    limit = 0.0
    for i, tool in enumerate(others):
        pair = wide[[reference, tool]].dropna()
        x = pair[reference].to_numpy()
        y = pair[tool].to_numpy()
        limit = max(limit, float(np.abs(np.concatenate([x, y])).max()))
        ax_f.scatter(
            x,
            y,
            s=1.6,
            color=color_of[tool],
            alpha=0.55,
            linewidths=0,
            zorder=3,
            label=display_tool(tool),
        )
        panel_f[f"{tool}_vs_{reference}"] = {
            "n": int(x.size),
            "pearson_r": float(np.corrcoef(x, y)[0, 1]) if x.size > 1 else None,
            "frac_same_sign": float(np.mean(np.sign(x) == np.sign(y))),
            "median_abs_difference": float(np.median(np.abs(x - y))),
            "median_abs_x": float(np.median(np.abs(x))),
            "median_abs_y": float(np.median(np.abs(y))),
        }
    bound = limit * 1.05
    ax_f.plot([-bound, bound], [-bound, bound], color="#666666", linewidth=0.4)
    ax_f.axhline(0.0, color="#CACACA", linewidth=0.3, zorder=0)
    ax_f.axvline(0.0, color="#CACACA", linewidth=0.3, zorder=0)
    ax_f.set_xscale("symlog", linthresh=1.0, linscale=0.6)
    ax_f.set_yscale("symlog", linthresh=1.0, linscale=0.6)
    ax_f.set_xlim(-bound, bound)
    ax_f.set_ylim(-bound, bound)
    for axis in (ax_f.xaxis, ax_f.yaxis):
        axis.set_ticks([-10, -1, 0, 1, 10])
        axis.set_ticklabels(["-10", "-1", "0", "1", "10"])
    ax_f.set_xlabel(f"\u0394 dGf'0 {display_tool(reference)} (kJ/mol)")
    ax_f.set_ylabel("\u0394 dGf'0 other tool (kJ/mol)")
    _panel_letter(ax_f, "f")
    panels.append(
        PanelNumbers(
            panel="f",
            what=(
                "per-compound signed dGf'0 shifts of each tool against"
                f" {reference}, both in kJ/mol"
            ),
            values=panel_f,
        )
    )

    # One legend for the whole figure, in the top margin, outside every axes.
    handles = [
        Patch(
            facecolor=color_of[t],
            edgecolor="black",
            linewidth=0.4,
            label=display_tool(t),
        )
        for t in tools
    ]
    handles.append(Line2D([0], [0], color="#666666", linewidth=0.4, label="identity"))
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(left / FIG_W_MM, 1.0),
        ncol=len(handles),
        frameon=True,
        fancybox=False,
        edgecolor="black",
        borderpad=0.3,
        handlelength=1.2,
        columnspacing=1.0,
    )

    os.makedirs(IMAGE_DIR, exist_ok=True)
    savefig_true_size_svg(fig, f"{FIGURE_STEM}.svg")
    fig.savefig(f"{FIGURE_STEM}.png", dpi=600)
    plt.close(fig)

    wide_out = wide.reindex(columns=tools)
    meta = (
        df[["compound_id", "inchi_key", "stratum", "groups", "n_cache_pkas"]]
        .drop_duplicates(subset="compound_id")
        .set_index("compound_id")
    )
    shift = meta.join(wide_out, how="right").reset_index()
    shift.to_csv(SHIFT_CSV, index=False)

    numbers = FigureNumbers(
        built_at=_timestamp(),
        source_csv=args.csv,
        figure_svg=f"{FIGURE_STEM}.svg",
        figure_png=f"{FIGURE_STEM}.png",
        conditions={"p_h": P_H, "min_ph": MIN_PH, "max_ph": MAX_PH},
        tools_drawn=tools,
        tools_excluded=excluded,
        panels=panels,
    )
    with open(FIGURE_JSON, "w") as handle:
        handle.write(numbers.model_dump_json(indent=1))
    print(f"wrote {FIGURE_STEM}.svg and .png")
    print(f"wrote {FIGURE_JSON}")
    print(f"wrote {SHIFT_CSV}")
    print(f"tools drawn: {tools}; excluded for n < {MIN_ROWS_FOR_PANEL}: {excluded}")
    assert FIG_H_MM <= MAX_HEIGHT_MM, "figure taller than the Nature page allows"


if __name__ == "__main__":
    main()
