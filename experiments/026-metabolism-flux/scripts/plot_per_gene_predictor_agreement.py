# experiments/026-metabolism-flux/scripts/plot_per_gene_predictor_agreement.py
# [[experiments.026-metabolism-flux.scripts.plot_per_gene_predictor_agreement]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/026-metabolism-flux/scripts/plot_per_gene_predictor_agreement.py

r"""How far apart the predictors are for ONE GENE, which is the unit a model consumes.

The agreement table in ``compare_kinetic_predictors.py`` is keyed by
(enzyme, substrate) PAIR. That is the right key for asking whether two models predict the
same number for the same measurement, but it is not the key the flux layer reads: a
catalytic unit draws one :math:`k_{cat}` per gene, so the question that decides whether a
predicted table is usable is how much the choice of predictor moves THAT gene's value.

A gene can carry several units and several substrates, so the per-gene value is the median
over its pairs in :math:`\log_{10}` space, taken separately for each predictor. Median
rather than mean because the pair distribution is heavy-tailed and one lipid substrate
should not set a gene's turnover number.

FOUR PANELS
-------------
a. The across-predictor spread per gene, in decades. This is the error bar on a single
   gene's :math:`k_{cat}` when the predictor is chosen arbitrarily.
b. The same spread laid against the dynamic range of :math:`k_{cat}` itself, so the
   spread can be compared to the signal rather than read in isolation.
c. Gene-level rank agreement between predictors. Pair-level Spearman is already reported;
   this asks the different question of whether two predictors order the GENES alike.
d. The genes that feed the betaxanthin cassette. Two of the four cassette members are
   heterologous and have no reaction in yeast-GEM, so no predictor emits a value for them;
   that absence is drawn rather than left out.
"""

import argparse
import json
import os
import os.path as osp
from typing import cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from matplotlib.colors import LinearSegmentedColormap

from torchcell.timestamp import timestamp
from torchcell.utils import (
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    apply_paper_style,
    mm_to_in,
    savefig_true_size_svg,
)

load_dotenv()
DATA_ROOT = cast(str, os.getenv("DATA_ROOT"))
ASSET_IMAGES_DIR = cast(str, os.getenv("ASSET_IMAGES_DIR"))
EXPERIMENT_ROOT = cast(str, os.getenv("EXPERIMENT_ROOT"))
KINETICS = osp.join(DATA_ROOT, "data", "torchcell", "kinetics")
RESULTS = osp.join(EXPERIMENT_ROOT, "026-metabolism-flux", "results")
OUT_DIR = osp.join(ASSET_IMAGES_DIR, "026-metabolism-flux")

DISPLAY = {
    "dlkcat": "DLKcat",
    "unikp": "UniKP",
    "eitlem": "EITLEM",
    "turnup": "TurNuP",
    "deepenzyme": "DeepEnzyme",
    "boost_km": "Boost_KM",
}

# The Btx-cassette as torchcell.datasets.scerevisiae.cachera2023 declares it, plus the
# aromatic-amino-acid genes immediately upstream that supply its tyrosine precursor.
# CYP76AD1 and DOD are the heterologous members: they are named here so their ABSENCE
# from every predictor table is visible in panel d rather than silently dropped.
CASSETTE: list[tuple[str, str]] = [
    ("CYP76AD1", "CYP76AD1"),
    ("DOD", "DOD"),
    ("YBR249C", "ARO4"),
    ("YPR060C", "ARO7"),
]
UPSTREAM: list[tuple[str, str]] = [
    ("YDR035W", "ARO3"),
    ("YDR127W", "ARO1"),
    ("YGL148W", "ARO2"),
    ("YBR166C", "TYR1"),
    ("YGL202W", "ARO8"),
    ("YHR137W", "ARO9"),
    ("YNL316C", "PHA2"),
]


def per_gene_log10(parameter: str) -> pd.DataFrame:
    """Gene x predictor table of median log10 values, from every built parquet."""
    columns: dict[str, pd.Series] = {}
    for predictor in sorted(os.listdir(KINETICS)):
        path = osp.join(KINETICS, predictor, "processed", f"{parameter}.parquet")
        if not osp.exists(path):
            continue
        frame = pd.read_parquet(path)
        values = frame[frame[parameter] > 0].copy()
        values["log10"] = np.log10(values[parameter])
        columns[predictor] = values.groupby("gene_id")["log10"].median()
    return pd.DataFrame(columns)


def box(ax: plt.Axes) -> None:
    """All four spines visible, the repo's boxed look."""
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)


def main() -> None:
    """Draw the four panels and record every number they show."""
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

    table = per_gene_log10(parameter)
    predictors = list(table.columns)
    complete = table.dropna()
    summary: dict[str, object] = {
        "parameter": parameter,
        "predictors": predictors,
        "n_genes_any": int(len(table)),
        "n_genes_all_predictors": int(len(complete)),
    }

    fig, axes = plt.subplots(
        2, 2, figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(105)), dpi=300
    )

    # --- a. the across-predictor spread for one gene ---------------------------
    ax = axes[0, 0]
    spread = (complete.max(axis=1) - complete.min(axis=1)).to_numpy()
    ax.hist(spread, bins=45, color=PLOT_PALETTE[0], edgecolor="black", linewidth=0.3)
    ax.axvline(float(np.median(spread)), color="black", linewidth=0.8, linestyle="--")
    ax.text(
        float(np.median(spread)),
        ax.get_ylim()[1] * 0.94,
        f"  median {np.median(spread):.2f}",
        fontsize=5,
        va="top",
    )
    ax.set_xlabel(f"across-predictor spread of {symbol} (decades)")
    ax.set_ylabel("genes")
    box(ax)
    summary["spread_median_decades"] = float(np.median(spread))
    summary["spread_p90_decades"] = float(np.percentile(spread, 90))
    summary["spread_frac_above_1_decade"] = float((spread > 1.0).mean())

    # --- b. spread against the dynamic range of the quantity itself ------------
    ax = axes[0, 1]
    center = complete.median(axis=1).to_numpy()
    order = np.argsort(center)
    ax.fill_between(
        np.arange(len(order)),
        complete.min(axis=1).to_numpy()[order],
        complete.max(axis=1).to_numpy()[order],
        color=PLOT_PALETTE[4],
        alpha=0.45,
        linewidth=0,
        label="predictor min-max",
    )
    ax.plot(
        np.arange(len(order)),
        center[order],
        color="black",
        linewidth=0.7,
        label="median over predictors",
    )
    ax.set_xlabel("gene, ranked by median across predictors")
    ax.set_ylabel(f"log$_{{10}}$ {symbol}")
    ax.legend(loc="upper left", frameon=False, fontsize=5, handlelength=1.4,
              handletextpad=0.4, borderpad=0.1)
    box(ax)
    # The comparison the panel exists to make: is the disagreement between instruments
    # larger than the range of the quantity they are measuring?
    signal = float(np.percentile(center, 90) - np.percentile(center, 10))
    summary["central_p10_p90_range_decades"] = signal
    summary["spread_over_signal"] = float(np.median(spread) / signal)

    # --- c. gene-level rank agreement -----------------------------------------
    ax = axes[1, 0]
    n = len(predictors)
    grid = np.full((n, n), np.nan)
    for i, left in enumerate(predictors):
        for j, right in enumerate(predictors):
            if i == j:
                # A predictor against itself is 1 by construction and carries no
                # information, so the diagonal is left blank rather than drawn as the
                # brightest cell on the panel.
                continue
            both = table[[left, right]].dropna()
            if len(both) >= 3:
                grid[i, j] = both[left].corr(both[right], method="spearman")
    # Sequential map built from the repo palette rather than a matplotlib default, so the
    # panel stays inside the green-free scheme: blue (low) through sand to orange (high).
    agreement_cmap = LinearSegmentedColormap.from_list(
        "tc_agreement", [PLOT_PALETTE[4], PLOT_PALETTE[15], PLOT_PALETTE[0]]
    )
    agreement_cmap.set_bad("#FFFFFF")
    image = ax.imshow(np.ma.masked_invalid(grid), cmap=agreement_cmap, vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_xticklabels([DISPLAY.get(p, p) for p in predictors], rotation=35, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels([DISPLAY.get(p, p) for p in predictors])
    for i in range(n):
        for j in range(n):
            if np.isfinite(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center", fontsize=5,
                        color="white" if grid[i, j] < 0.35 else "black")
    bar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    bar.set_label("gene-level Spearman", fontsize=6)
    bar.ax.tick_params(labelsize=5, width=0.5)
    bar.outline.set_linewidth(0.5)
    box(ax)
    summary["gene_level_spearman"] = {
        f"{predictors[i]}|{predictors[j]}": (
            None if not np.isfinite(grid[i, j]) else float(grid[i, j])
        )
        for i in range(n)
        for j in range(n)
        if i < j
    }

    # --- d. the genes the betaxanthin case study runs on -----------------------
    ax = axes[1, 1]
    wanted = CASSETTE + UPSTREAM
    labels, offsets = [], []
    absent: list[str] = []
    for row, (orf, common) in enumerate(wanted):
        labels.append(common)
        offsets.append(row)
        if orf not in table.index:
            absent.append(common)
            ax.text(0.5, row, "no reaction in yeast-GEM", fontsize=5, style="italic",
                    color="#666666", va="center", ha="center",
                    transform=ax.get_yaxis_transform())
            continue
        for k, predictor in enumerate(predictors):
            value = table.loc[orf, predictor]
            if np.isfinite(value):
                ax.scatter(value, row, s=9, color=PLOT_PALETTE[k % len(PLOT_PALETTE)],
                           edgecolors="black", linewidths=0.3, zorder=3,
                           label=DISPLAY.get(predictor, predictor) if row == 2 else None)
    ax.set_yticks(offsets)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.axhline(len(CASSETTE) - 0.5, color="black", linewidth=0.5, linestyle=":")
    ax.set_xlabel(f"log$_{{10}}$ {symbol}")
    ax.set_ylabel("Btx cassette (top), tyrosine supply (bottom)")
    left_limit, right_limit = ax.get_xlim()
    ax.set_xlim(left_limit, right_limit + 0.42 * (right_limit - left_limit))
    handles, names = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, names, loc="center right", frameon=False, fontsize=5,
                  handletextpad=0.2, borderpad=0.1, labelspacing=0.25)
    box(ax)
    summary["cassette_genes_absent_from_all_predictors"] = absent

    for letter, ax in zip("abcd", axes.flatten()):
        ax.text(-0.16, 1.06, letter, transform=ax.transAxes, fontsize=8,
                fontweight="bold", va="top")

    fig.tight_layout()
    base = f"per_gene_agreement_{parameter}{stamp}"
    savefig_true_size_svg(fig, osp.join(OUT_DIR, f"{base}.svg"))
    fig.savefig(osp.join(OUT_DIR, f"{base}.png"), dpi=300)
    plt.close(fig)

    os.makedirs(RESULTS, exist_ok=True)
    with open(osp.join(RESULTS, f"per_gene_agreement_{parameter}.json"), "w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"figure written to {osp.join(OUT_DIR, base)}.svg")


if __name__ == "__main__":
    main()
