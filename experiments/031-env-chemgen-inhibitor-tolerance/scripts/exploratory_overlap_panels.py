# experiments/031-env-chemgen-inhibitor-tolerance/scripts/exploratory_overlap_panels.py
# [[experiments.031-env-chemgen-inhibitor-tolerance.scripts.exploratory_overlap_panels]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/031-env-chemgen-inhibitor-tolerance/scripts/exploratory_overlap_panels
"""Where the five datasets overlap and where they do not: genotype, chemistry, dose.

The joinability argument needs the overlap drawn on every axis a pooled model sees, because
each axis fails differently. Genes overlap heavily and compounds barely. Chemical space
splits into a drug-like region and a small-molecule region the corpora do not cover.
Concentration is stated on three incompatible bases.

FOUR PANEL GROUPS, all read from the served records and the precomputed embeddings.

* **Genotype.** Gene-set overlap between datasets, and the perturbation classes each carries.
  The gene axis is where pooling is cheapest: the panels show how much of one dataset's gene
  universe another already covers.
* **Chemistry.** Molecular weight against calculated logP for every dosed compound, which
  separates the drug-like partner libraries from the small hydrolysate molecules; the heavy
  atom count distribution; and each Vanacloig compound's nearest neighbor in the partner
  union under Tanimoto on count ECFP4.
* **Dose.** The molar range each dataset spans, with the compounds that carry no number at
  all marked, and the spread of the doses given to compounds that appear in more than one
  dataset.
* **Cells.** How the measured (gene, compound) cells distribute over the gene and compound
  axes, which is what says whether a dataset is wide and shallow or narrow and deep.

Reads ``results/records_*.parquet``, ``results/embeddings/*.npz`` and the RDKit descriptor
block written by ``embed_compounds.py``. Writes ``results/chemical_property_table.csv`` and
``results/genotype_overlap.csv``, and the figure ``exploratory_overlap.{svg,png}`` into
``ASSET_IMAGES_DIR/031-env-chemgen-inhibitor-tolerance/``.
"""

from __future__ import annotations

import os
import os.path as osp

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from torchcell.molecule.similarity import tanimoto_matrix
from torchcell.utils import PLOT_PALETTE, mm_to_in, savefig_true_size_svg

load_dotenv()
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
ASSET_IMAGES_DIR = os.environ["ASSET_IMAGES_DIR"]
RESULTS_DIR = osp.join(
    EXPERIMENT_ROOT, "031-env-chemgen-inhibitor-tolerance", "results"
)
EMB_DIR = osp.join(RESULTS_DIR, "embeddings")
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
HOST_GENES = {"YBL005W", "YDR011W", "YGL013C"}
TO_MOLAR = {"M": 1.0, "mM": 1e-3, "uM": 1e-6, "nM": 1e-9, "pM": 1e-12}
#: The RDKit 2D descriptor block's column order is the RDKit descriptor list; these are the
#: two the chemistry panels use, located by name at load time rather than by a fixed index.
DESCRIPTORS = ("MolWt", "MolLogP", "HeavyAtomCount")


def queried_gene(genotype: str) -> str:
    """The one perturbed gene of a genotype that is not part of the sensitizing host."""
    genes = [g for g in str(genotype).split("|") if g not in HOST_GENES]
    return genes[0] if len(genes) == 1 else ""


def load_axes(name: str) -> pd.DataFrame:
    """Single-gene single-compound records with their gene, compound key and dose."""
    df = pd.read_parquet(
        osp.join(RESULTS_DIR, f"records_{name}.parquet"),
        columns=[
            "gene",
            "inchikey",
            "perturbation_type",
            "n_small_molecules",
            "dose_value",
            "dose_unit",
            "dose_basis",
        ],
    )
    df = df[df["n_small_molecules"] == 1]
    df["qgene"] = df["gene"].map(queried_gene)
    df = df[(df["qgene"] != "") & (df["inchikey"].astype(str).str.len() > 0)]
    df["dataset"] = name
    return df


def molar(df: pd.DataFrame) -> pd.Series:
    """Dose in molar where the unit converts without a molecular weight, else NA."""
    val = pd.to_numeric(df["dose_value"], errors="coerce")
    factor = df["dose_unit"].map(TO_MOLAR)
    return val * factor


def descriptor_table(keys: list[str]) -> pd.DataFrame:
    """Molecular weight, logP and heavy atom count for every key, from the RDKit block."""
    from rdkit.Chem import Descriptors

    names = [d[0] for d in Descriptors._descList]
    cols = {d: names.index(d) for d in DESCRIPTORS if d in names}
    z = np.load(osp.join(EMB_DIR, "rdkit_2d.npz"), allow_pickle=True)
    idx = {str(k): i for i, k in enumerate(z["inchikey"])}
    X = np.asarray(z["X"], dtype=np.float64)
    rows = []
    for k in keys:
        if k not in idx:
            continue
        row: dict[str, object] = {"inchikey": k}
        for d, c in cols.items():
            row[d] = float(X[idx[k], c])
        rows.append(row)
    return pd.DataFrame(rows).set_index("inchikey")


def nearest_neighbor_to_panel(panel: list[str], partners: list[str]) -> pd.DataFrame:
    """Each panel compound's best non-exact Tanimoto to the partner union, count ECFP4."""
    z = np.load(osp.join(EMB_DIR, "ecfp4_count.npz"), allow_pickle=True)
    idx = {str(k): i for i, k in enumerate(z["inchikey"])}
    X = np.asarray(z["X"], dtype=np.float64)
    pa = [k for k in panel if k in idx]
    pb = [k for k in partners if k in idx]
    sims = tanimoto_matrix(X[[idx[k] for k in pa]], X[[idx[k] for k in pb]])
    rows = []
    for i, k in enumerate(pa):
        s = sims[i].copy()
        for j, k2 in enumerate(pb):
            if k2 == k:
                s[j] = np.nan  # an exact match is not a neighbor
        rows.append(
            {
                "inchikey": k,
                "nn": float(np.nanmax(s)) if np.isfinite(s).any() else float("nan"),
                "exact": k in set(pb),
            }
        )
    return pd.DataFrame(rows)


def panel_gene_overlap(ax: plt.Axes, axes_by_ds: dict[str, pd.DataFrame]) -> None:
    """Share of the row dataset's genes that the column dataset also measures."""
    genes = {n: set(axes_by_ds[n]["qgene"]) for n in NAMES}
    n = len(NAMES)
    M = np.zeros((n, n))
    for i, a in enumerate(NAMES):
        for j, b in enumerate(NAMES):
            M[i, j] = len(genes[a] & genes[b]) / max(len(genes[a]), 1)
    im = ax.imshow(M, cmap="YlOrBr", vmin=0, vmax=1)
    for i in range(n):
        for j in range(n):
            ax.text(
                j,
                i,
                f"{M[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=4.5,
                color="black" if M[i, j] < 0.6 else "white",
            )
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([LABEL[x] for x in NAMES], rotation=45, ha="right")
    ax.set_yticklabels([f"{LABEL[x]} ({len(genes[x]):,})" for x in NAMES])
    ax.set_title("a  row's genes also in column", loc="left", fontsize=6)
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.035, pad=0.02, shrink=0.8)
    cb.ax.tick_params(labelsize=4.5)


def panel_compound_overlap(ax: plt.Axes, axes_by_ds: dict[str, pd.DataFrame]) -> None:
    """The same for compounds, which is where pooling stops being cheap."""
    cpds = {n: set(axes_by_ds[n]["inchikey"]) for n in NAMES}
    n = len(NAMES)
    M = np.zeros((n, n))
    for i, a in enumerate(NAMES):
        for j, b in enumerate(NAMES):
            M[i, j] = len(cpds[a] & cpds[b]) / max(len(cpds[a]), 1)
    im = ax.imshow(M, cmap="YlOrBr", vmin=0, vmax=1)
    for i in range(n):
        for j in range(n):
            ax.text(
                j,
                i,
                f"{M[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=4.5,
                color="black" if M[i, j] < 0.6 else "white",
            )
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([LABEL[x] for x in NAMES], rotation=45, ha="right")
    ax.set_yticklabels([f"{LABEL[x]} ({len(cpds[x]):,})" for x in NAMES])
    ax.set_title("b  row's compounds also in column", loc="left", fontsize=6)
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.035, pad=0.02, shrink=0.8)
    cb.ax.tick_params(labelsize=4.5)


def panel_chemical_space(
    ax: plt.Axes, keys_by_ds: dict[str, set[str]], desc: pd.DataFrame
) -> None:
    """Molecular weight against logP: the drug-like region and the small-molecule region."""
    for name in ("wildenhain2015", "hillenmeyer2008_het", "hoepfner2014"):
        k = [x for x in keys_by_ds[name] if x in desc.index]
        ax.scatter(
            desc.loc[k, "MolWt"],
            desc.loc[k, "MolLogP"],
            s=1.5,
            color=COLOR[name],
            alpha=0.35,
            linewidths=0,
            label=LABEL[name],
        )
    k = [x for x in keys_by_ds["vanacloig2022"] if x in desc.index]
    ax.scatter(
        desc.loc[k, "MolWt"],
        desc.loc[k, "MolLogP"],
        s=9,
        color=PLOT_PALETTE[0],
        edgecolor="black",
        lw=0.5,
        zorder=4,
        label="Vanacloig",
    )
    ax.set_xlim(0, 700)
    ax.set_xlabel("molecular weight")
    ax.set_ylabel("calculated logP")
    ax.legend(frameon=False, fontsize=4.5, loc="lower right", scatterpoints=1)
    ax.set_title("c  the panel is small and polar", loc="left", fontsize=6)


def panel_heavy_atoms(
    ax: plt.Axes, keys_by_ds: dict[str, set[str]], desc: pd.DataFrame
) -> None:
    """Heavy atom count: why a substructure fingerprint is nearly empty for the alcohols."""
    bins = np.arange(0, 61, 2)
    for name in NAMES:
        k = [x for x in keys_by_ds[name] if x in desc.index]
        v = desc.loc[k, "HeavyAtomCount"].to_numpy()
        ax.hist(
            v,
            bins=bins,
            density=True,
            histtype="step",
            color=COLOR[name],
            lw=0.9,
            label=LABEL[name],
        )
    ax.axvspan(0, 8, color="#F5F5F5", zorder=0)
    ax.set_xlabel("heavy atom count")
    ax.set_ylabel("density")
    ax.legend(frameon=False, fontsize=4.5, loc="upper right")
    # placed after the legend so the axes limits are settled, and low so the two cannot
    # land on top of each other
    ax.annotate(
        "ECFP nearly empty",
        (8.5, ax.get_ylim()[1] * 0.45),
        fontsize=4.5,
        ha="left",
        color="#666666",
    )
    ax.set_title("d  corpora are drug-like", loc="left", fontsize=6)


def panel_nearest_neighbor(ax: plt.Axes, nn: pd.DataFrame, desc: pd.DataFrame) -> None:
    """Each Vanacloig compound's nearest partner, against its size."""
    m = nn.set_index("inchikey").join(desc, how="inner")
    ax.scatter(
        m["HeavyAtomCount"],
        m["nn"],
        s=10,
        color=[PLOT_PALETTE[1] if e else PLOT_PALETTE[0] for e in m["exact"]],
        edgecolor="black",
        lw=0.4,
        zorder=3,
    )
    ax.axhline(0.5, color="#666666", lw=0.5, ls="--")
    ax.set_xlabel("heavy atom count")
    ax.set_ylabel("best non-exact Tanimoto in the partners")
    ax.set_ylim(0, 1)
    ax.set_title("e  small compounds, no neighbor", loc="left", fontsize=6)


def panel_dose(ax: plt.Axes, axes_by_ds: dict[str, pd.DataFrame]) -> None:
    """The molar range each dataset spans, and what fraction has no number at all."""
    y = np.arange(len(NAMES))
    for i, name in enumerate(NAMES):
        df = axes_by_ds[name]
        m = molar(df).dropna()
        m = m[m > 0]
        frac = len(m) / max(len(df), 1)
        if len(m) == 0:
            ax.text(
                -9.5, i, "no convertible dose", fontsize=5, va="center", color="#A24A46"
            )
            continue
        lo, hi = np.log10(m.quantile(0.01)), np.log10(m.quantile(0.99))
        ax.plot([lo, hi], [i, i], color=COLOR[name], lw=2.5, solid_capstyle="round")
        ax.scatter([np.log10(m.median())], [i], s=12, color="black", zorder=4)
        ax.text(hi + 0.2, i, f"{100 * frac:.0f}% stated", fontsize=4.5, va="center")
    ax.set_yticks(y)
    ax.set_yticklabels([LABEL[n] for n in NAMES])
    ax.invert_yaxis()
    ax.set_xlim(-10.5, 3)
    ax.set_xlabel("log10 molar, 1st to 99th percentile")
    ax.set_title("f  dose does not pool", loc="left", fontsize=6)


def panel_cells(ax: plt.Axes, axes_by_ds: dict[str, pd.DataFrame]) -> None:
    """Genes against compounds, point area the measured cells: shape of each dataset."""
    # Hoepfner and Hillenmeyer HET sit almost on top of each other (5,839 against 5,810
    # genes), so their labels are pushed to opposite sides rather than both to the right
    offsets = {
        "vanacloig2022": (7, 0),
        "hillenmeyer2008_hom": (7, 0),
        "hillenmeyer2008_het": (7, 7),
        "hoepfner2014": (-8, -4),
        "wildenhain2015": (-7, 0),
    }
    for name in NAMES:
        df = axes_by_ds[name]
        ng, nc = df["qgene"].nunique(), df["inchikey"].nunique()
        cells = len(df.drop_duplicates(["qgene", "inchikey"]))
        ax.scatter(
            nc,
            ng,
            s=8 + 12 * np.log10(max(cells, 10)),
            color=COLOR[name],
            edgecolor="black",
            lw=0.5,
            zorder=3,
        )
        dx, dy = offsets[name]
        ax.annotate(
            f"{LABEL[name]}\n{cells:,} cells",
            (nc, ng),
            textcoords="offset points",
            xytext=(dx, dy),
            fontsize=4.5,
            va="center",
            ha="right" if dx < 0 else "left",
        )
    ax.set_xscale("log")
    ax.set_xlim(20, 20000)
    ax.set_ylim(0, 7200)
    ax.set_xlabel("distinct compounds (log)")
    ax.set_ylabel("distinct genes")
    ax.set_title("g  dataset shape", loc="left", fontsize=6)


def panel_perturbation(ax: plt.Axes, axes_by_ds: dict[str, pd.DataFrame]) -> None:
    """The perturbation classes each dataset contributes, as a share of its records."""

    def collapse(kind: str) -> str:
        """A multi-gene genotype is named by its class, not by its marker permutation.

        Vanacloig writes the same four-deletion strain in four marker orders, which are one
        perturbation class and would otherwise fill the legend with permutations.
        """
        parts = str(kind).split("|")
        if len(parts) > 1:
            return f"{len(parts)}x deletion (barcoded + markers)"
        return parts[0].replace("_", " ")

    classes: dict[str, dict[str, float]] = {}
    for name in NAMES:
        vc = (
            axes_by_ds[name]["perturbation_type"]
            .map(collapse)
            .value_counts(normalize=True)
        )
        classes[name] = vc.to_dict()
    kinds = sorted({k for d in classes.values() for k in d})
    bottom = np.zeros(len(NAMES))
    for c, kind in enumerate(kinds):
        vals = np.array([classes[n].get(kind, 0.0) for n in NAMES])
        ax.bar(
            np.arange(len(NAMES)),
            vals,
            bottom=bottom,
            color=PLOT_PALETTE[c % len(PLOT_PALETTE)],
            edgecolor="black",
            lw=0.4,
            label=kind.replace("_", " "),
        )
        bottom += vals
    ax.set_xticks(np.arange(len(NAMES)))
    ax.set_xticklabels([LABEL[n] for n in NAMES], rotation=45, ha="right")
    ax.set_ylabel("share of records")
    ax.set_ylim(0, 1)
    ax.set_title("h  perturbation class by dataset", loc="left", fontsize=6)


def main() -> None:
    axes_by_ds = {n: load_axes(n) for n in NAMES}
    keys_by_ds = {n: set(axes_by_ds[n]["inchikey"]) for n in NAMES}
    for n in NAMES:
        print(
            f"{n}: {len(axes_by_ds[n]):,} records, {axes_by_ds[n]['qgene'].nunique():,} "
            f"genes, {len(keys_by_ds[n]):,} compounds"
        )

    all_keys = sorted(set().union(*keys_by_ds.values()))
    desc = descriptor_table(all_keys)
    desc.to_csv(osp.join(RESULTS_DIR, "chemical_property_table.csv"))
    print(f"descriptors for {len(desc):,} of {len(all_keys):,} compounds")

    partners = sorted(set().union(*[keys_by_ds[n] for n in NAMES[1:]]))
    nn = nearest_neighbor_to_panel(sorted(keys_by_ds["vanacloig2022"]), partners)
    print(
        f"Vanacloig nearest neighbors: median {nn['nn'].median():.3f}, "
        f"{int((nn['nn'] > 0.5).sum())} above 0.5, {int(nn['exact'].sum())} exact"
    )

    rows = []
    for n in NAMES:
        df = axes_by_ds[n]
        rows.append(
            {
                "dataset": n,
                "records": len(df),
                "genes": df["qgene"].nunique(),
                "compounds": df["inchikey"].nunique(),
                "cells": len(df.drop_duplicates(["qgene", "inchikey"])),
                "perturbation_types": "|".join(
                    sorted(set(df["perturbation_type"].astype(str)))
                ),
            }
        )
    pd.DataFrame(rows).to_csv(
        osp.join(RESULTS_DIR, "genotype_overlap.csv"), index=False
    )

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
            "hatch.linewidth": 0.4,
        }
    )
    fig, axes = plt.subplots(
        3, 3, figsize=(mm_to_in(179.0), mm_to_in(155.0)), constrained_layout=True
    )
    panel_gene_overlap(axes[0, 0], axes_by_ds)
    panel_compound_overlap(axes[0, 1], axes_by_ds)
    panel_chemical_space(axes[0, 2], keys_by_ds, desc)
    panel_heavy_atoms(axes[1, 0], keys_by_ds, desc)
    panel_nearest_neighbor(axes[1, 1], nn, desc)
    panel_dose(axes[1, 2], axes_by_ds)
    panel_cells(axes[2, 0], axes_by_ds)
    panel_perturbation(axes[2, 1], axes_by_ds)
    # the ninth panel carries panel h's legend rather than a frame, which keeps the legend
    # off the bars and out of constrained_layout's way
    handles, labels = axes[2, 1].get_legend_handles_labels()
    axes[2, 2].legend(
        handles,
        labels,
        frameon=False,
        fontsize=5,
        loc="center left",
        title="h  classes",
    )
    axes[2, 2].axis("off")
    for ax in axes.ravel():
        for side in ("top", "right", "bottom", "left"):
            ax.spines[side].set_visible(True)
    os.makedirs(IMAGE_DIR, exist_ok=True)
    # the raster goes FIRST: savefig_true_size_svg rescales the figure from 72 dpi to
    # draw.io's 100 units per inch, and a PNG written after it inherits that rescaling,
    # which shrinks every axes against type that is still sized in points
    fig.savefig(osp.join(IMAGE_DIR, "exploratory_overlap.png"), dpi=300)
    savefig_true_size_svg(fig, osp.join(IMAGE_DIR, "exploratory_overlap.svg"))
    plt.close(fig)
    print(f"  wrote {osp.join(IMAGE_DIR, 'exploratory_overlap.svg')}")


if __name__ == "__main__":
    main()
