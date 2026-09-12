# experiments/028-knockout-expression/scripts/nadal_identify_deletion.py
# [[experiments.028-knockout-expression.scripts.nadal_identify_deletion]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/028-knockout-expression/scripts/nadal_identify_deletion
"""Can the deleted gene be read off a genotype's own expression profile?

A deletion removes its own transcript, so in a clean knockout profile the deleted gene is
the most reduced gene, or close to it. This is the one fact about a profile that needs
no second study to check, and it tests the gene mapping and the cell-to-genotype
assignment at once: if the labeled cells are the genotype, its deleted gene is at the
bottom of the profile; if they are mostly other genotypes, it is not.

For every single-deletion genotype, the rank of its own deleted gene among all genes in
its profile, ascending (rank 1 = the most reduced gene), as a fraction of genes ranked,
and the fraction of genotypes with the deleted gene at rank 1, in the bottom 10, and in
the bottom 1%. Three profiles per genotype:

  Kemmeren 2014           the microarray log2 ratio (the reference behavior)
  Nadal stored            scanpy logfoldchanges as stored, sentinels dropped
  Nadal A pseudobulk      log2((cpm_g + 1)/(cpm_wt + 1)) from raw UMI (recompute script)
  Nadal B Seurat          FindMarkers avg_log2FC from the same cells

Kemmeren's own recommended Perturb-seq self-check (Replogle 2022 style) is the same
statistic; the paper does not report it.

Run from the repo root after nadal_pseudobulk_recompute.R:
    python experiments/028-knockout-expression/scripts/nadal_identify_deletion.py
"""

from __future__ import annotations

import json
import os
import os.path as osp
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from matplotlib.colors import to_rgba  # noqa: E402

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from cross_study_ko_expression import (  # noqa: E402
    LMDB,
    _is_control,
    _load_lmdb,
    _profiles,
)
from cross_study_recomputed import (  # noqa: E402
    RECOMPUTED_REL,
    _recomputed_profiles,
    _resolver,
)

from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome  # noqa: E402
from torchcell.utils import (  # noqa: E402
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    mm_to_in,
    panel_label,
    savefig_true_size_svg,
)
from torchcell.utils.paths import experiment_results_dir  # noqa: E402


def _self_rank(profiles: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Per genotype: the deleted gene's value, its ascending rank and the genes ranked."""
    rows = []
    for orf, prof in profiles.items():
        v = prof.get(orf)
        if v is None:
            rows.append((orf, np.nan, np.nan, len(prof), False))
            continue
        vals = np.fromiter(prof.values(), dtype=float)
        rank = int((vals < v).sum()) + 1
        rows.append((orf, float(v), rank, len(vals), True))
    return pd.DataFrame(
        rows, columns=["orf", "self_value", "self_rank", "n_genes", "self_present"]
    )


def _summary(df: pd.DataFrame) -> dict[str, Any]:
    d = df[df["self_present"]]
    frac = d["self_rank"] / d["n_genes"]
    return {
        "n_genotypes": int(len(df)),
        "n_self_present": int(len(d)),
        "median_self_value": float(d["self_value"].median()),
        "frac_self_below_minus1": float((d["self_value"] < -1).mean()),
        "median_rank": float(d["self_rank"].median()),
        "median_rank_fraction": float(frac.median()),
        "frac_rank_1": float((d["self_rank"] == 1).mean()),
        "frac_rank_le_10": float((d["self_rank"] <= 10).mean()),
        "frac_rank_bottom_1pct": float((frac <= 0.01).mean()),
        "frac_rank_bottom_10pct": float((frac <= 0.10).mean()),
    }


def main() -> None:
    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    images = osp.join(os.environ["ASSET_IMAGES_DIR"], "028-knockout-expression")
    os.makedirs(images, exist_ok=True)
    results_dir = experiment_results_dir("028-knockout-expression", __file__)

    genome = SCerevisiaeGenome(
        genome_root=osp.join(data_root, "data/sgd/genome"),
        go_root=osp.join(data_root, "data/go"),
        overwrite=False,
    )
    resolve = _resolver(genome)
    genotypes = pd.read_csv(
        osp.join(data_root, RECOMPUTED_REL, "genotypes.tsv"), sep="\t"
    )

    kem, _, _ = _profiles(_load_lmdb(osp.join(data_root, LMDB["kemmeren"])))
    nad_records = [
        r for r in _load_lmdb(osp.join(data_root, LMDB["nadal"])) if _is_control(r)
    ]
    stored, _, _ = _profiles(nad_records)
    versions: dict[str, dict[str, dict[str, float]]] = {
        "kemmeren": kem,
        "nadal_stored": stored,
    }
    cells: dict[str, int] = {}
    for stat in ("pseudobulk_log2fc", "seurat_avg_log2fc"):
        versions[f"nadal_{stat}"], cells, _ = _recomputed_profiles(
            data_root, stat, resolve, genotypes
        )

    tables = {k: _self_rank(v) for k, v in versions.items()}
    summary = {k: _summary(t) for k, t in tables.items()}
    for k, s in summary.items():
        print(
            f"{k:26s} n {s['n_self_present']:5d}/{s['n_genotypes']:5d}  self value median "
            f"{s['median_self_value']:6.2f}  < -1: {100 * s['frac_self_below_minus1']:5.1f}%  "
            f"rank median {s['median_rank']:6.0f} ({100 * s['median_rank_fraction']:.1f}th pct)  "
            f"rank 1: {100 * s['frac_rank_1']:5.1f}%  bottom 10: {100 * s['frac_rank_le_10']:5.1f}%  "
            f"bottom 1%: {100 * s['frac_rank_bottom_1pct']:5.1f}%"
        )

    # Does the self-rank in Nadal A improve with cells behind the genotype?
    t = tables["nadal_pseudobulk_log2fc"].copy()
    t["cells"] = t["orf"].map(cells)
    d = t[t["self_present"] & t["cells"].notna()]
    by_cells: dict[str, Any] = {}
    for lo, hi in ((0, 30), (30, 100), (100, 300), (300, 10**6)):
        m = (d["cells"] >= lo) & (d["cells"] < hi)
        if m.sum():
            by_cells[f"{lo}-{hi}"] = {
                "n": int(m.sum()),
                "frac_bottom_1pct": float(
                    ((d.loc[m, "self_rank"] / d.loc[m, "n_genes"]) <= 0.01).mean()
                ),
                "median_self_value": float(d.loc[m, "self_value"].median()),
            }
    print("Nadal A by cells per genotype:", json.dumps(by_cells))

    out = {
        "generated_by": "experiments/028-knockout-expression/scripts/nadal_identify_deletion.py",
        "summary": summary,
        "nadal_pseudobulk_by_cells": by_cells,
    }
    with open(osp.join(results_dir, "nadal_identify_deletion.json"), "w") as f:
        json.dump(out, f, indent=1)
    pd.concat(
        [t.assign(version=k) for k, t in tables.items()], ignore_index=True
    ).to_csv(
        osp.join(results_dir, "nadal_identify_deletion_per_genotype.csv"), index=False
    )

    # ------------------------------------------------------------------ figure
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 6,
            "axes.linewidth": 0.5,
            "svg.fonttype": "none",
            "axes.titlesize": 6,
            "legend.fontsize": 6,
        }
    )
    order = [
        "kemmeren",
        "nadal_stored",
        "nadal_pseudobulk_log2fc",
        "nadal_seurat_avg_log2fc",
    ]
    names = {
        "kemmeren": "Kemmeren",
        "nadal_stored": "Nadal stored",
        "nadal_pseudobulk_log2fc": "Nadal A pseudobulk",
        "nadal_seurat_avg_log2fc": "Nadal B Seurat",
    }
    col = dict(
        zip(order, [PLOT_PALETTE[3], PLOT_PALETTE[1], PLOT_PALETTE[0], PLOT_PALETTE[2]])
    )
    legend_kw = dict(frameon=True, edgecolor="black", fancybox=False, framealpha=1.0)
    fig, axes = plt.subplots(
        1, 3, figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(52))
    )

    # a. the deleted gene's own value
    ax = axes[0]
    bins = np.linspace(-6, 3, 55)
    for k in order:
        d = tables[k][tables[k]["self_present"]]
        ax.hist(
            d["self_value"].clip(-6, 3),
            bins=bins,
            histtype="stepfilled",
            facecolor=to_rgba(col[k], 0.45),
            edgecolor="black",
            lw=0.4,
            density=True,
            label=f"{names[k]} (med {d['self_value'].median():.2f})",
        )
    ax.axvline(0, color="black", lw=0.5, ls="--")
    ax.set_xlabel("deleted gene's own log2 value")
    ax.set_ylabel("density")
    ax.set_title("the deleted gene in its own profile")
    # Headroom so the framed legend sits above the tallest histogram.
    ax.set_ylim(0, ax.get_ylim()[1] * 1.9)
    ax.legend(loc="upper right", **legend_kw)

    # b. rank fraction, cumulative
    ax = axes[1]
    for k in order:
        d = tables[k][tables[k]["self_present"]]
        frac = np.sort((d["self_rank"] / d["n_genes"]).to_numpy())
        ax.plot(
            frac,
            np.arange(1, len(frac) + 1) / len(frac),
            color=col[k],
            lw=1.0,
            label=names[k],
        )
    ax.plot([0, 1], [0, 1], color="black", lw=0.5, ls="--", label="chance")
    ax.set_xscale("log")
    ax.set_xlim(1e-4, 1)
    ax.set_xlabel("rank of the deleted gene, ascending, as a fraction of genes")
    ax.set_ylabel("cumulative fraction of genotypes")
    ax.set_title("where the deleted gene ranks")
    ax.legend(loc="upper left", **legend_kw)

    # c. Nadal A: self value against cells per genotype
    ax = axes[2]
    t = tables["nadal_pseudobulk_log2fc"].copy()
    t["cells"] = t["orf"].map(cells)
    d = t[t["self_present"] & t["cells"].notna()]
    ax.scatter(
        d["cells"],
        d["self_value"],
        s=3,
        color=col["nadal_pseudobulk_log2fc"],
        lw=0,
        alpha=0.6,
    )
    ax.axhline(0, color="black", lw=0.5, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("cells per genotype")
    ax.set_ylabel("deleted gene's own log2 FC (A)")
    ax.set_title("Nadal A: self value vs cell count")

    for ax in axes:
        for s in ax.spines.values():
            s.set_visible(True)
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.17, top=0.86, wspace=0.35)
    for ax, letter in zip(axes, "abc"):
        panel_label(ax, letter)
    stem = osp.join(images, "nadal_identify_deletion")
    fig.savefig(stem + ".png", dpi=300)
    savefig_true_size_svg(fig, stem + ".svg")
    print(f"\nfigure: {stem}.svg")
    print(f"results: {osp.join(results_dir, 'nadal_identify_deletion.json')}")


if __name__ == "__main__":
    main()
