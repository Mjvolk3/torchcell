# experiments/031-env-chemgen-inhibitor-tolerance/scripts/dataset_joinability.py
# [[experiments.031-env-chemgen-inhibitor-tolerance.scripts.dataset_joinability]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/031-env-chemgen-inhibitor-tolerance/scripts/dataset_joinability
"""Can these chemogenomic datasets be pooled into one training target?

Pooling is an assertion about the LABEL, not only about the input: two records that measure
the same gene under the same compound must mean the same thing before a single loss can score
them. This script tests that assertion on the served records and writes the evidence.

FOUR QUESTIONS, EACH MEASURED.

1. **Scale.** What is each dataset's response distribution? If the standard deviations differ
   by a large factor, a pooled squared-error loss is dominated by the widest dataset
   regardless of how many records the others contribute.

2. **Polarity.** Does a more negative number mean a sicker strain in every dataset? It does
   not. Each loader records its source's own definition verbatim, and those definitions split
   the five datasets into two groups with OPPOSITE sign conventions (``POLARITY`` below). The
   script does not take that on faith: it predicts the sign of every pairwise correlation from
   the declared polarities and checks the prediction against the measurement.

3. **Overlap.** How many (gene, compound) cells are measured by more than one dataset? This is
   the only place where agreement can be observed at all, and it is what decides whether a
   shared response scale is testable rather than assumed.

4. **Agreement.** On those shared cells, do the two datasets correlate once oriented? This is
   the joinability evidence. A near-zero oriented correlation means the datasets can share an
   INPUT representation and a gene prior while still needing their own output scale.

A record enters only if it doses exactly one compound with an InChIKey and perturbs exactly
one gene, so a cell is a (gene, compound) pair and nothing is averaged across combinations.
Vanacloig's three host deletions are stripped first, leaving the queried gene.

Writes ``results/dataset_distributions.csv``, ``results/cross_dataset_pair_overlap.csv``,
``results/polarity_check.csv``, and the figure ``dataset_joinability.{svg,png}`` into
``ASSET_IMAGES_DIR/031-env-chemgen-inhibitor-tolerance/``.
"""

from __future__ import annotations

import os
import os.path as osp
from itertools import combinations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy.stats import binomtest, pearsonr, skew, spearmanr

from torchcell.utils import PLOT_PALETTE, mm_to_in, savefig_true_size_svg

load_dotenv()
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
ASSET_IMAGES_DIR = os.environ["ASSET_IMAGES_DIR"]
RESULTS_DIR = osp.join(
    EXPERIMENT_ROOT, "031-env-chemgen-inhibitor-tolerance", "results"
)
IMAGE_DIR = osp.join(ASSET_IMAGES_DIR, "031-env-chemgen-inhibitor-tolerance")

NAMES = [
    "vanacloig2022",
    "hillenmeyer2008_hom",
    "hillenmeyer2008_het",
    "hoepfner2014",
    "wildenhain2015",
]
LABEL = {
    "vanacloig2022": "Vanacloig",
    "hillenmeyer2008_hom": "Hillenmeyer HOM",
    "hillenmeyer2008_het": "Hillenmeyer HET",
    "hoepfner2014": "Hoepfner",
    "wildenhain2015": "Wildenhain",
}
COLOR = dict(zip(NAMES, PLOT_PALETTE[:5], strict=True))
# the drug-sensitized host carried by every Vanacloig genotype on top of the queried deletion
HOST_GENES = {"YBL005W", "YDR011W", "YGL013C"}

#: Sign convention of each served response, quoted from the source through its loader. The
#: value is the sign a SICK strain carries, so multiplying the response by it puts every
#: dataset on one orientation where NEGATIVE means a fitness defect.
#:
#: These are the sources' own words, not an inference from the data. The empirical check
#: below is independent: it predicts every pairwise correlation sign from this table and
#: compares against the measured sign.
POLARITY: dict[str, dict[str, object]] = {
    "vanacloig2022": {
        "sick_sign": -1,
        "measurement": "log2_ratio",
        "quote": "log2((CPM of the inhibitor replicate + 1) / (mean CPM of the SAME CG "
        "batch's inhibitor-free controls + 1)); depletion under the inhibitor is negative",
    },
    "hillenmeyer2008_hom": {
        "sick_sign": +1,
        "measurement": "z_score",
        "quote": "HOP fitness-defect z-score, (mean control intensity - treatment "
        "intensity) / SD of the control intensities; positive = fitness defect",
    },
    "hillenmeyer2008_het": {
        "sick_sign": +1,
        "measurement": "log2_ratio",
        "quote": "HIP fitness-defect log-ratio, log2(mean control intensity / treatment "
        "intensity) averaged over the up and down tags; positive = fitness defect",
    },
    "hoepfner2014": {
        "sick_sign": -1,
        "measurement": "sensitivity_score",
        "quote": "adjusted MADL sensitivity score ...; negative = hypersensitive, "
        "positive = resistant; 0 = growth equal to control",
    },
    "wildenhain2015": {
        "sick_sign": -1,
        "measurement": "z_score",
        "quote": "released PubChem AID 1159580 z_score of growth inhibition ...; "
        "negative = growth inhibition",
    },
}


def queried_gene(genotype: str) -> str:
    """The one perturbed gene of a genotype that is not part of the sensitizing host."""
    genes = [g for g in str(genotype).split("|") if g not in HOST_GENES]
    return genes[0] if len(genes) == 1 else ""


def load_cells(name: str) -> pd.Series:
    """Mean response per (gene, InChIKey) cell for the single-gene single-compound records.

    Averaging over the doses and screens of a cell is deliberate: the question here is
    whether two DATASETS agree about a gene under a compound, and a per-dose comparison
    would confound that with the dose disagreement measured separately.
    """
    df = pd.read_parquet(
        osp.join(RESULTS_DIR, f"records_{name}.parquet"),
        columns=["gene", "inchikey", "response", "n_small_molecules"],
    )
    df = df[df["n_small_molecules"] == 1]
    df["qgene"] = df["gene"].map(queried_gene)
    df = df[(df["qgene"] != "") & (df["inchikey"].astype(str).str.len() > 0)]
    return df.groupby(["qgene", "inchikey"])["response"].mean()


def distribution_rows(cells: dict[str, pd.Series]) -> pd.DataFrame:
    """Per dataset: the response distribution, its skew, and the declared polarity."""
    rows = []
    for name in NAMES:
        v = cells[name].to_numpy()
        pol = POLARITY[name]
        rows.append(
            {
                "dataset": name,
                "measurement_type": pol["measurement"],
                "sick_sign": pol["sick_sign"],
                "cells": len(v),
                "mean": float(np.mean(v)),
                "sd": float(np.std(v, ddof=1)),
                "median": float(np.median(v)),
                "q01": float(np.quantile(v, 0.01)),
                "q99": float(np.quantile(v, 0.99)),
                "skew": float(skew(v)),
                # the tail that carries the phenotype: a sick strain is an outlier, so the
                # heavier tail should lie on the side the declared polarity names
                "tail_ratio": float(
                    abs(np.quantile(v, 0.01) - np.median(v))
                    / max(abs(np.quantile(v, 0.99) - np.median(v)), 1e-12)
                ),
            }
        )
    return pd.DataFrame(rows)


def overlap_rows(cells: dict[str, pd.Series]) -> pd.DataFrame:
    """Every dataset pair: shared genes, compounds and cells, raw and oriented agreement."""
    rows = []
    for a, b in combinations(NAMES, 2):
        ca, cb = cells[a], cells[b]
        genes = set(ca.index.get_level_values(0)) & set(cb.index.get_level_values(0))
        cpds = set(ca.index.get_level_values(1)) & set(cb.index.get_level_values(1))
        joined = ca.to_frame("a").join(cb.to_frame("b"), how="inner")
        n = len(joined)
        if n >= 20:
            rho = float(spearmanr(joined["a"], joined["b"])[0])
            r = float(pearsonr(joined["a"], joined["b"])[0])
        else:
            rho, r = float("nan"), float("nan")
        # the orientation factor is +1 when the two sources already agree about which sign
        # means sick, and -1 when they do not
        orient = int(POLARITY[a]["sick_sign"]) * int(POLARITY[b]["sick_sign"])
        rows.append(
            {
                "a": a,
                "b": b,
                "shared_genes": len(genes),
                "shared_compounds": len(cpds),
                "shared_cells": n,
                "spearman_raw": rho,
                "pearson_raw": r,
                "orientation": orient,
                "spearman_oriented": rho * orient,
                "pearson_oriented": r * orient,
                "sign_predicted": orient,
                "sign_observed": int(np.sign(rho)) if np.isfinite(rho) else 0,
                "sign_agrees": bool(np.isfinite(rho) and np.sign(rho) == orient),
            }
        )
    return pd.DataFrame(rows)


def panel_distributions(ax: plt.Axes, cells: dict[str, pd.Series]) -> None:
    """Response distributions on a shared axis, which is what makes the scale gap visible."""
    for i, name in enumerate(NAMES):
        v = cells[name].to_numpy()
        v = v[np.isfinite(v)]
        lo, hi = np.quantile(v, [0.005, 0.995])
        grid = np.linspace(lo, hi, 256)
        hist, edges = np.histogram(v, bins=grid, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        ax.plot(centers, hist, color=COLOR[name], lw=0.9, label=LABEL[name])
    ax.set_xlim(-12, 12)
    # log density, because every distribution is sharply peaked at zero and the PHENOTYPE
    # lives in the tail; on a linear axis the peaks hide exactly what distinguishes them
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 5)
    ax.set_xlabel("served response (native units)")
    ax.set_ylabel("density (log)")
    ax.legend(frameon=False, fontsize=4.5, loc="lower center", ncol=2)
    ax.set_title("a  five different scales", loc="left", fontsize=6)


def panel_sd(ax: plt.Axes, dist: pd.DataFrame) -> None:
    """Standard deviation per dataset: the factor a pooled squared-error loss would see."""
    y = np.arange(len(dist))
    ax.barh(
        y,
        dist["sd"],
        color=[COLOR[n] for n in dist["dataset"]],
        edgecolor="black",
        lw=0.4,
    )
    ax.set_yticks(y)
    ax.set_yticklabels([LABEL[n] for n in dist["dataset"]])
    ax.invert_yaxis()
    ax.set_xlabel("standard deviation of the served response")
    ratio = dist["sd"].max() / dist["sd"].min()
    ax.set_title(f"b  widest / narrowest = {ratio:.0f}x", loc="left", fontsize=6)


def panel_polarity(ax: plt.Axes, dist: pd.DataFrame) -> None:
    """Skew beside the declared sick-sign: an independent check on the loaders' quotes.

    A sick strain is an outlier, so the heavy tail must fall on the side the source says
    means a fitness defect. Hatching marks the two datasets whose sources declare the
    opposite sign from the other three.
    """
    d = dist.sort_values("skew")
    y = np.arange(len(d))
    ax.barh(
        y,
        d["skew"],
        color=[COLOR[n] for n in d["dataset"]],
        edgecolor="black",
        lw=0.4,
        hatch=["///" if s > 0 else "" for s in d["sick_sign"]],
    )
    ax.set_yticks(y)
    ax.set_yticklabels(
        [
            f"{LABEL[n]} ({'+' if s > 0 else '-'})"
            for n, s in zip(d["dataset"], d["sick_sign"], strict=True)
        ]
    )
    ax.axvline(0, color="black", lw=0.5)
    ax.set_xlabel("skew of the response")
    ax.set_title("c  tail falls on the declared side", loc="left", fontsize=6)


def panel_matrix(
    ax: plt.Axes, ov: pd.DataFrame, value: str, title: str, fmt: str, log: bool
) -> None:
    """One lower-triangular dataset-by-dataset matrix."""
    n = len(NAMES)
    M = np.full((n, n), np.nan)
    for _, row in ov.iterrows():
        i, j = NAMES.index(row["a"]), NAMES.index(row["b"])
        M[max(i, j), min(i, j)] = row[value]
    shown = np.log10(np.where(M > 0, M, np.nan)) if log else M
    vmax = np.nanmax(np.abs(shown)) if np.isfinite(shown).any() else 1.0
    im = ax.imshow(
        shown,
        cmap="RdBu_r" if not log else "YlOrBr",
        vmin=-vmax if not log else None,
        vmax=vmax,
    )
    for i in range(n):
        for j in range(n):
            if np.isfinite(M[i, j]):
                ax.text(
                    j, i, format(M[i, j], fmt), ha="center", va="center", fontsize=4.5
                )
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([LABEL[x] for x in NAMES], rotation=45, ha="right")
    ax.set_yticklabels([LABEL[x] for x in NAMES])
    ax.set_title(title, loc="left", fontsize=6)
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cb.ax.tick_params(labelsize=4.5)


def panel_agreement(ax: plt.Axes, ov: pd.DataFrame) -> None:
    """Raw against oriented agreement, one point per pair, sized by shared cells."""
    big = ov[ov["shared_cells"] >= 20]
    ax.axhline(0, color="black", lw=0.4, ls=":")
    ax.axvline(0, color="black", lw=0.4, ls=":")
    for _, row in big.iterrows():
        ax.scatter(
            row["spearman_raw"],
            row["spearman_oriented"],
            s=6 + 10 * np.log10(max(row["shared_cells"], 10)),
            color=PLOT_PALETTE[0] if row["orientation"] > 0 else PLOT_PALETTE[1],
            edgecolor="black",
            lw=0.4,
            zorder=3,
        )
    lim = 1.15 * float(np.nanmax(np.abs(big[["spearman_raw"]].to_numpy())))
    ax.plot([-lim, lim], [-lim, lim], color="#666666", lw=0.4, ls="--")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("Spearman as served")
    ax.set_ylabel("Spearman after orientation")
    ax.set_title("f  orienting lifts every pair", loc="left", fontsize=6)


def make_figure(
    cells: dict[str, pd.Series], dist: pd.DataFrame, ov: pd.DataFrame
) -> None:
    """The six-panel joinability figure."""
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 6,
            "axes.labelsize": 6,
            "axes.titlesize": 6,
            "xtick.labelsize": 5,
            "ytick.labelsize": 5,
            "axes.linewidth": 0.5,
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(
        2, 3, figsize=(mm_to_in(179.0), mm_to_in(105.0)), constrained_layout=True
    )
    panel_distributions(axes[0, 0], cells)
    panel_sd(axes[0, 1], dist)
    panel_polarity(axes[0, 2], dist)
    panel_matrix(axes[1, 0], ov, "shared_cells", "d  shared cells", ",.0f", True)
    panel_matrix(
        axes[1, 1], ov, "spearman_oriented", "e  oriented agreement", ".3f", False
    )
    panel_agreement(axes[1, 2], ov)
    for ax in axes.ravel():
        for side in ("top", "right", "bottom", "left"):
            ax.spines[side].set_visible(True)
    os.makedirs(IMAGE_DIR, exist_ok=True)
    # the raster goes FIRST: savefig_true_size_svg rescales the figure from 72 dpi to
    # draw.io's 100 units per inch, and a PNG written after it inherits that rescaling,
    # which shrinks every axes against type that is still sized in points
    fig.savefig(osp.join(IMAGE_DIR, "dataset_joinability.png"), dpi=300)
    savefig_true_size_svg(fig, osp.join(IMAGE_DIR, "dataset_joinability.svg"))
    plt.close(fig)
    print(f"  wrote {osp.join(IMAGE_DIR, 'dataset_joinability.svg')}")


def main() -> None:
    cells = {name: load_cells(name) for name in NAMES}
    for name in NAMES:
        print(f"{name}: {len(cells[name]):,} single-gene single-compound cells")

    dist = distribution_rows(cells)
    dist.to_csv(osp.join(RESULTS_DIR, "dataset_distributions.csv"), index=False)
    print()
    print(dist.to_string(index=False))

    ov = overlap_rows(cells)
    ov.to_csv(osp.join(RESULTS_DIR, "cross_dataset_pair_overlap.csv"), index=False)
    print()
    print(
        ov[
            [
                "a",
                "b",
                "shared_genes",
                "shared_compounds",
                "shared_cells",
                "spearman_raw",
                "orientation",
                "spearman_oriented",
                "sign_agrees",
            ]
        ].to_string(index=False)
    )

    testable = ov[ov["shared_cells"] >= 20]
    agree = int(testable["sign_agrees"].sum())
    check = pd.DataFrame(
        [
            {
                "pairs_testable": len(testable),
                "sign_predictions_correct": agree,
                # under the null that the declared polarity says nothing about the data,
                # each pairwise sign is a coin flip
                "binomial_p_two_sided": float(
                    binomtest(agree, len(testable), 0.5).pvalue
                ),
                "median_spearman_oriented": float(
                    testable["spearman_oriented"].median()
                ),
                "max_spearman_oriented": float(testable["spearman_oriented"].max()),
            }
        ]
    )
    check.to_csv(osp.join(RESULTS_DIR, "polarity_check.csv"), index=False)
    print()
    print(
        f"polarity predicts {agree} of {len(testable)} pairwise correlation signs; "
        f"oriented Spearman median {testable['spearman_oriented'].median():.3f}, "
        f"max {testable['spearman_oriented'].max():.3f}"
    )

    make_figure(cells, dist, ov)


if __name__ == "__main__":
    main()
