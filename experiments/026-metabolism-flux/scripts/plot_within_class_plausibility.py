# experiments/026-metabolism-flux/scripts/plot_within_class_plausibility.py
# [[experiments.026-metabolism-flux.scripts.plot_within_class_plausibility]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/026-metabolism-flux/scripts/plot_within_class_plausibility.py

r"""How many of a predictor's points fall inside the within-class spread, by model.

The companion of the per-gene agreement figure: its panel e draws one shaded band and says
in the caption what fraction of each predictor's points sit inside it. These five panels make
that number the subject, and ask it for the betaxanthin genes specifically, which have no
measured yeast value and so can only be checked for class plausibility.

FIVE PANELS, three rows
-----------------------
a. Fraction of the measured pairs inside the shared band, as the class is read to fewer EC
   digits. The band at each depth is the median within-class standard deviation over all
   classes at that depth. The dashed line is the EC-median lookup, which knows the enzyme's
   function and nothing else.
b. The same fraction at EC4, against the fraction inside each pair's OWN class standard
   deviation. Wilson 95 percent intervals on both.
c. The betaxanthin supply genes, one column per (gene, reaction) unit. The gray band is one
   standard deviation about the median of the unit's EC4 class across the Open Enzyme
   Database, the black tick the class median, and the dots are the predictors. A unit with no
   band has an EC class with fewer than three measured enzymes, or no EC number.
d. Fraction of those units inside their class band, by predictor.
e. The half width of each unit's class band, so a high fraction in d can be read against how
   wide the target was.

Reads ``results/within_class_plausibility.json`` and the betaxanthin CSV beside it. One color
per predictor, the same map as every other figure in this experiment. Run from the repo root
with the torchcell env, after ``within_class_plausibility.py``.
"""

import argparse
import json
import os
import os.path as osp
import sys
from typing import cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from matplotlib.colors import to_rgba
from matplotlib.ticker import MultipleLocator

from torchcell.timestamp import timestamp
from torchcell.utils import (
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    PLOT_PALETTE_FILL,
    apply_paper_style,
    mm_to_in,
    savefig_true_size_svg,
)

load_dotenv()
ASSET_IMAGES_DIR = cast(str, os.getenv("ASSET_IMAGES_DIR"))
EXPERIMENT_ROOT = cast(str, os.getenv("EXPERIMENT_ROOT"))
RESULTS = osp.join(EXPERIMENT_ROOT, "026-metabolism-flux", "results")
OUT_DIR = osp.join(ASSET_IMAGES_DIR, "026-metabolism-flux")

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from plot_per_gene_predictor_agreement import (  # noqa: E402
    BAND_SLOT,
    DISPLAY,
    PREDICTOR_COLOR_INDEX,
    predictor_color,
)

LEVELS = ["ec1", "ec2", "ec3", "ec4"]


def box(ax: plt.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)


def legend_above(ax: plt.Axes, ncol: int, handles=None, names=None) -> None:
    if handles is None:
        handles, names = ax.get_legend_handles_labels()
    ax.legend(
        handles, names, loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=ncol,
        frameon=False, fontsize=6, handlelength=1.2, handletextpad=0.4, borderpad=0.0,
        columnspacing=1.0,
    )


def tenths(ax: plt.Axes) -> None:
    """A gridline every 0.1 on a 0 to 1 axis, labeled every 0.2."""
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.tick_params(axis="y", which="minor", length=0)
    ax.grid(axis="y", which="both", linewidth=0.3, color="0.85")
    ax.set_axisbelow(True)


def whiskers(fraction: dict) -> tuple[float, float]:
    return fraction["frac"] - fraction["ci95"][0], fraction["ci95"][1] - fraction["frac"]


def panel_a(ax: plt.Axes, block: dict, ordered: list[str]) -> None:
    positions = np.arange(len(LEVELS))
    for predictor in ordered:
        levels = block["shared_band"][predictor]
        ax.plot(
            positions, [levels[k]["frac"] for k in LEVELS], marker="o", markersize=3,
            color=predictor_color(predictor), linewidth=1.0, markeredgecolor="black",
            markeredgewidth=0.3, label=DISPLAY[predictor],
        )
    baseline = block["ec_median_lookup"]["shared"]
    ax.plot(
        positions, [baseline[k]["frac"] for k in LEVELS], color="black", linewidth=0.8,
        linestyle="--", label="EC-median lookup",
    )
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [f"EC{k[-1]}\n±{block['band_half_width_by_level'][k]:.2f}" for k in LEVELS]
    )
    ax.set_xlabel("EC digits defining the class, and the band half width")
    ax.set_ylabel("fraction of measured pairs\ninside the shared band")
    tenths(ax)
    legend_above(ax, ncol=4)
    box(ax)


def panel_b(ax: plt.Axes, block: dict, ordered: list[str]) -> None:
    positions = np.arange(len(ordered))
    width = 0.38
    shared = [block["shared_band"][p]["ec4"] for p in ordered]
    own = [block["own_class_band"]["predictors"][p] for p in ordered]
    ax.bar(
        positions - width / 2, [f["frac"] for f in shared], width,
        color=[predictor_color(p) for p in ordered], edgecolor="black", linewidth=0.4,
        yerr=np.array([whiskers(f) for f in shared]).T,
        error_kw={"elinewidth": 0.5, "capsize": 1.2, "capthick": 0.5, "ecolor": "black"},
        label="shared EC4 band",
    )
    # The lighter member of a two-level bar takes the fill color, which is the one use the
    # palette reserves it for.
    ax.bar(
        positions + width / 2, [f["frac"] for f in own], width,
        color=[PLOT_PALETTE_FILL[PREDICTOR_COLOR_INDEX[p] % len(PLOT_PALETTE_FILL)]
               for p in ordered],
        edgecolor="black", linewidth=0.4,
        yerr=np.array([whiskers(f) for f in own]).T,
        error_kw={"elinewidth": 0.5, "capsize": 1.2, "capthick": 0.5, "ecolor": "black"},
        label="own EC4 class band",
    )
    baseline = block["ec_median_lookup"]
    ax.axhline(baseline["shared"]["ec4"]["frac"], color="black", linewidth=0.7,
               linestyle="--")
    ax.set_xticks(positions)
    ax.set_xticklabels([DISPLAY[p] for p in ordered], rotation=35, ha="right")
    ax.set_ylabel("fraction of measured pairs\ninside the band")
    tenths(ax)
    handles, names = ax.get_legend_handles_labels()
    handles.append(plt.Line2D([0], [0], color="black", linewidth=0.7, linestyle="--"))
    names.append("EC-median lookup, shared band")
    legend_above(ax, ncol=2, handles=handles, names=names)
    box(ax)


def unit_label(row: pd.Series) -> str:
    ec = row["ec"] if isinstance(row["ec"], str) else "no EC"
    return f"{row['common_name']} {row['reaction_id'].replace('r_', 'r')}\n{ec}"


def panel_c(ax: plt.Axes, units: pd.DataFrame, ordered: list[str], symbol: str) -> None:
    positions = np.arange(len(units))
    scorable = units["class_sd_log10"].notna().to_numpy()
    lower = (units["class_median_log10"] - units["class_sd_log10"]).to_numpy()
    upper = (units["class_median_log10"] + units["class_sd_log10"]).to_numpy()
    ax.bar(
        positions[scorable], (upper - lower)[scorable], 0.7, bottom=lower[scorable],
        # The class band is a reference REGION, not a series, so it takes the muted purple
        # no predictor uses (slot 15, dusty mauve) rather than the gray slot EZKinetics is
        # drawn in. Solid fill rather than a 25 percent alpha on a line color: alpha makes a
        # tone the palette does not define and changes with whatever sits behind it.
        color=PLOT_PALETTE_FILL[BAND_SLOT], linewidth=0, zorder=1,
        label="EC4 class, median ± 1 s.d.",
    )
    ax.hlines(
        units["class_median_log10"].to_numpy()[scorable], positions[scorable] - 0.35,
        positions[scorable] + 0.35, color="black", linewidth=0.8, zorder=2,
    )
    for predictor in ordered:
        values = units["predicted_log10"].map(lambda d: d.get(predictor, np.nan)).to_numpy()
        ax.scatter(
            positions, values, s=9, color=predictor_color(predictor), edgecolors="black",
            linewidths=0.3, zorder=3, label=DISPLAY[predictor],
        )
    ax.set_xticks(positions)
    ax.set_xticklabels([unit_label(r) for _, r in units.iterrows()], rotation=90)
    ax.set_xlim(-0.6, len(units) - 0.4)
    ax.set_ylabel(f"log$_{{10}}$ {symbol}")
    legend_above(ax, ncol=7)
    box(ax)


def panel_d(ax: plt.Axes, block: dict, ordered: list[str]) -> None:
    per = block["betaxanthin"]["per_predictor_inside_one_sd"]
    positions = np.arange(len(ordered))
    fractions = [per[p] for p in ordered]
    ax.bar(
        positions, [f["frac"] for f in fractions], 0.7,
        color=[predictor_color(p) for p in ordered], edgecolor="black", linewidth=0.4,
        yerr=np.array([whiskers(f) for f in fractions]).T,
        error_kw={"elinewidth": 0.5, "capsize": 1.2, "capthick": 0.5, "ecolor": "black"},
    )
    ax.set_xticks(positions)
    ax.set_xticklabels([DISPLAY[p] for p in ordered], rotation=35, ha="right")
    ax.set_ylabel("betaxanthin units inside\ntheir class band")
    tenths(ax)
    box(ax)


def panel_e(ax: plt.Axes, units: pd.DataFrame, block: dict) -> None:
    scorable = units[units["class_sd_log10"].notna()]
    positions = np.arange(len(scorable))
    # One series takes the FIRST primary, not the gray slot: gray is a secondary and enters
    # a series at its sixth color.
    ax.bar(positions, scorable["class_sd_log10"].to_numpy(), 0.7, color=PLOT_PALETTE[0],
           edgecolor="black", linewidth=0.4)
    ax.axhline(block["band_half_width_by_level"]["ec4"], color="black", linewidth=0.7,
               linestyle="--")
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [f"{r['common_name']} {r['reaction_id'].replace('r_', 'r')}"
         for _, r in scorable.iterrows()], rotation=90,
    )
    ax.set_ylabel("class band half width\n(orders of magnitude)")
    legend_above(
        ax, ncol=1,
        handles=[plt.Line2D([0], [0], color="black", linewidth=0.7, linestyle="--")],
        names=["shared EC4 band"],
    )
    box(ax)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parameter", default="k_cat", choices=["k_cat", "K_M"])
    parser.add_argument("--no-timestamp", action="store_true")
    args = parser.parse_args()

    apply_paper_style()
    plt.rcParams.update({"xtick.major.width": 0.5, "ytick.major.width": 0.5})
    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = "" if args.no_timestamp else f"_{timestamp()}"
    parameter = args.parameter
    symbol = r"$k_{cat}$" if parameter == "k_cat" else r"$K_M$"

    with open(osp.join(RESULTS, "within_class_plausibility.json")) as handle:
        report = json.load(handle)
    block = report["parameters"][parameter]
    ordered = sorted(
        block["shared_band"], key=lambda p: -block["shared_band"][p]["ec4"]["frac"]
    )
    units = pd.read_csv(osp.join(RESULTS, "within_class_plausibility_betaxanthin.csv"))
    units = units[units["parameter"] == parameter].reset_index(drop=True)
    units["predicted_log10"] = units["predicted_log10"].map(
        lambda s: json.loads(s.replace("'", '"'))
    )

    fig = plt.figure(figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(158)), dpi=300)
    grid = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.25, 1.0])
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, :])
    ax_d = fig.add_subplot(grid[2, 0])
    ax_e = fig.add_subplot(grid[2, 1])
    panel_a(ax_a, block, ordered)
    panel_b(ax_b, block, ordered)
    panel_c(ax_c, units, ordered, symbol)
    panel_d(ax_d, block, ordered)
    panel_e(ax_e, units, block)

    for letter, ax, dx in zip("abcde", (ax_a, ax_b, ax_c, ax_d, ax_e),
                              (-0.14, -0.14, -0.065, -0.14, -0.14)):
        ax.text(dx, 1.16, letter, transform=ax.transAxes, fontsize=8, fontweight="bold",
                va="top")

    fig.tight_layout(w_pad=2.6, h_pad=2.4)
    base = f"within_class_plausibility_{parameter}{stamp}"
    savefig_true_size_svg(fig, osp.join(OUT_DIR, f"{base}.svg"))
    fig.savefig(osp.join(OUT_DIR, f"{base}.png"), dpi=300)
    plt.close(fig)
    print(f"figure -> {osp.join(OUT_DIR, base)}.svg")


if __name__ == "__main__":
    main()
