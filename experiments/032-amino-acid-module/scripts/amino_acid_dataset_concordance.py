# experiments/032-amino-acid-module/scripts/amino_acid_dataset_concordance.py
# [[experiments.032-amino-acid-module.scripts.amino_acid_dataset_concordance]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/032-amino-acid-module/scripts/amino_acid_dataset_concordance

r"""Do the two deletion-collection amino-acid screens agree? Mulleder 2016 against Cooper 2010.

Two genome-wide screens measured free amino acids in single-deletion strains and both are
built as torchcell datasets:

* Mulleder 2016 (``amino_acid_mulleder2016``): 19 amino acids by LC-SRM/MS, intracellular
  mM, the PROTOTROPHIC collection (markers restored) grown on synthetic minimal medium,
  one culture per strain; 4,678 strains. The target the 032 arms train on.
* Cooper 2010 (``amino_acid_cooper2010``): 17 CE-LIF peaks, each a linear ratio to its
  plate mean, the auxotrophic MATa collection grown overnight in synthetic complete medium
  (seven supplements absent); 4,313 strains kept, duplicates screened in the same run.
  Several peaks are composites (leucine with isoleucine and citrulline, glutamine with
  valine, methionine with proline, asparagine with tyrosine), lysine is two peaks, and
  four peaks have no Mulleder counterpart (a lysine-related peak, glutathione with
  biotin, N-acetylornithine, ornithine).

The two differ in strain background, medium, chemistry and normalization, so agreement is
a property of the biology that survives all four. This script reads both loaders'
``preprocess/data.csv`` tables (their build manifests are recorded in the output),
matches strains by systematic gene name, and reports, for every Cooper peak against every
Mulleder amino acid, the Spearman correlation over the strains both measured (a full
17 x 19 matrix, so a matched pair is read against the background of unmatched ones), the
Pearson correlation of the two log-scaled quantities, and for the matched pairs the
overlap of the top 5 percent of strains. A composite peak is matched to the SUM of its
components' mM and to each component. Two reliability references bound what a
cross-screen correlation can reach: Mulleder's own retest (194 strains measured twice,
``experiments/027-betaxanthin-metabolic/results/mulleder_retest_reliability.json``) and
Cooper's replicate strains (the 47 duplicate rows the loader keeps in its ledger, paired
with the served row). Writes ``results/amino_acid_dataset_concordance.json`` and a figure
(the Spearman matrix with the matched pairs outlined, and the matched pairs against both
retest references) under ``ASSET_IMAGES_DIR/032-amino-acid-module``. Run from the repo
root::

    PYTHONPATH=$PWD ~/miniconda3/envs/torchcell/bin/python \
        experiments/032-amino-acid-module/scripts/amino_acid_dataset_concordance.py
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
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Rectangle
from pydantic import BaseModel
from scipy.stats import hypergeom, pearsonr, spearmanr

from torchcell.timestamp import timestamp
from torchcell.utils import PANEL_WIDTHS_MM, PLOT_PALETTE, mm_to_in, savefig_true_size_svg

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
IMAGES = osp.join(os.environ["ASSET_IMAGES_DIR"], "032-amino-acid-module")
RESULTS = osp.join(EXPERIMENT_ROOT, "032-amino-acid-module", "results")
MULLEDER = osp.join(DATA_ROOT, "data/torchcell/amino_acid_mulleder2016")
COOPER = osp.join(DATA_ROOT, "data/torchcell/amino_acid_cooper2010")
RELIABILITY = osp.join(
    EXPERIMENT_ROOT, "027-betaxanthin-metabolic/results/mulleder_retest_reliability.json"
)
MULLEDER_KEYS = [
    "alanine", "arginine", "asparagine", "aspartate", "glutamate", "glutamine",
    "glycine", "histidine", "isoleucine", "leucine", "lysine", "methionine",
    "phenylalanine", "proline", "serine", "threonine", "tryptophan", "tyrosine",
    "valine",
]
COOPER_KEYS = [
    "lysine_related_peak1", "arginine", "gshbiotin", "n_acetylornithine",
    "leucine+isoleucine+citrulline", "glutamine+valine", "methionine+proline",
    "threonine", "alanine", "serine", "asparagine+tyrosine", "glycine", "lysine_a",
    "ornithine", "lysine_b", "glutamate", "aspartate",
]
#: Cooper peak -> the Mulleder amino acids it carries (a composite peak lists several).
PEAK_MAP: dict[str, list[str]] = {
    "arginine": ["arginine"],
    "threonine": ["threonine"],
    "alanine": ["alanine"],
    "serine": ["serine"],
    "glycine": ["glycine"],
    "glutamate": ["glutamate"],
    "aspartate": ["aspartate"],
    "lysine_a": ["lysine"],
    "lysine_b": ["lysine"],
    "leucine+isoleucine+citrulline": ["leucine", "isoleucine"],
    "glutamine+valine": ["glutamine", "valine"],
    "methionine+proline": ["methionine", "proline"],
    "asparagine+tyrosine": ["asparagine", "tyrosine"],
}
TOP_FRAC = 0.05
AA3 = {
    "alanine": "Ala", "arginine": "Arg", "asparagine": "Asn", "aspartate": "Asp",
    "glutamate": "Glu", "glutamine": "Gln", "glycine": "Gly", "histidine": "His",
    "isoleucine": "Ile", "leucine": "Leu", "lysine": "Lys", "methionine": "Met",
    "phenylalanine": "Phe", "proline": "Pro", "serine": "Ser", "threonine": "Thr",
    "tryptophan": "Trp", "tyrosine": "Tyr", "valine": "Val", "citrulline": "Cit",
    "lysine_a": "Lys a", "lysine_b": "Lys b",
}


def short(name: str) -> str:
    """Three-letter codes for the figure's tick labels; composites keep their plus."""
    return "+".join(AA3.get(part, part) for part in name.split("+"))


class PairStat(BaseModel):
    """One Cooper peak against one Mulleder quantity."""

    cooper_peak: str
    mulleder: str
    n: int
    spearman: float
    spearman_p: float
    pearson_log: float
    n_log: int
    top5_overlap: int
    top5_expected: float
    top5_enrichment: float
    top5_hypergeom_p: float


class Retest(BaseModel):
    key: str
    n_pairs: int
    spearman: float
    pearson_log: float


class Report(BaseModel):
    mulleder_manifest: dict[str, Any]
    cooper_manifest: dict[str, Any]
    n_mulleder: int
    n_cooper: int
    n_shared_strains: int
    peak_map: dict[str, list[str]]
    matched: list[PairStat]
    matrix_spearman: list[list[float | None]]
    matrix_n: list[list[int]]
    cooper_keys: list[str]
    mulleder_keys: list[str]
    mulleder_retest: dict[str, dict[str, float]]
    cooper_duplicate_retest: list[Retest]
    figure: str
    note: str


def manifest(root: str) -> dict[str, Any]:
    with open(osp.join(root, "preprocess", "build_manifest.json")) as fh:
        m = json.load(fh)
    return {k: v for k, v in m.items() if not isinstance(v, (dict, list))}


def pair_stat(
    peak: str, name: str, x: np.ndarray, y: np.ndarray, rng: np.random.Generator
) -> PairStat:
    """``x`` Cooper ratio, ``y`` Mulleder mM (or a sum), over strains both measured."""
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    rho, p = spearmanr(x, y)
    pos = (x > 0) & (y > 0)
    r_log = float(pearsonr(np.log10(x[pos]), np.log10(y[pos]))[0]) if pos.sum() > 3 else float("nan")
    k = max(1, int(round(TOP_FRAC * x.size)))
    top_x = set(np.argsort(-x)[:k].tolist())
    top_y = set(np.argsort(-y)[:k].tolist())
    overlap = len(top_x & top_y)
    expected = k * k / x.size
    return PairStat(
        cooper_peak=peak,
        mulleder=name,
        n=int(x.size),
        spearman=float(rho),
        spearman_p=float(p),
        pearson_log=r_log,
        n_log=int(pos.sum()),
        top5_overlap=overlap,
        top5_expected=float(expected),
        top5_enrichment=float(overlap / expected),
        top5_hypergeom_p=float(hypergeom.sf(overlap - 1, x.size, k, k)),
    )


def main() -> None:
    rng = np.random.default_rng(0)
    mul = pd.read_csv(osp.join(MULLEDER, "preprocess", "data.csv")).set_index("orf")
    coo = pd.read_csv(osp.join(COOPER, "preprocess", "data.csv"))
    assert set(mul.columns) == set(MULLEDER_KEYS), list(mul.columns)
    assert all(k in coo.columns for k in COOPER_KEYS), list(coo.columns)
    # Strains are matched on the collection's own ORF name: a renamed (merged) ORF is
    # served under its current systematic name with the source name kept, so two
    # strains can share a systematic name while naming different deletions.
    assert not coo["perturbed_gene_name"].duplicated().any()
    coo = coo.set_index("perturbed_gene_name")
    shared = sorted(set(mul.index) & set(coo.index))
    print(f"Mulleder {len(mul)} strains, Cooper {len(coo)}, shared {len(shared)}")
    m = mul.loc[shared, MULLEDER_KEYS].to_numpy(dtype=float)
    c = coo.loc[shared, COOPER_KEYS].to_numpy(dtype=float)

    # The full matrix: every peak against every amino acid.
    matrix = np.full((len(COOPER_KEYS), len(MULLEDER_KEYS)), np.nan)
    counts = np.zeros(matrix.shape, dtype=int)
    for i in range(len(COOPER_KEYS)):
        for j in range(len(MULLEDER_KEYS)):
            ok = np.isfinite(c[:, i]) & np.isfinite(m[:, j])
            counts[i, j] = int(ok.sum())
            if ok.sum() >= 30:
                matrix[i, j] = spearmanr(c[ok, i], m[ok, j]).correlation

    # The matched pairs: composite peaks against the sum and against each component.
    matched: list[PairStat] = []
    for peak, names in PEAK_MAP.items():
        x = c[:, COOPER_KEYS.index(peak)]
        if len(names) > 1:
            y = m[:, [MULLEDER_KEYS.index(n) for n in names]].sum(axis=1)
            matched.append(pair_stat(peak, "+".join(names), x, y, rng))
        for n in names:
            matched.append(pair_stat(peak, n, x, m[:, MULLEDER_KEYS.index(n)], rng))
    for s in matched:
        print(
            f"  {s.cooper_peak:<30s} vs {s.mulleder:<22s} n {s.n:4d} rho {s.spearman:+.3f} "
            f"(p {s.spearman_p:.1e}) log-Pearson {s.pearson_log:+.3f} top5 overlap "
            f"{s.top5_overlap}/{s.top5_expected:.1f} expected (x{s.top5_enrichment:.1f})"
        )

    # Mulleder's own retest, from the 027 read.
    with open(RELIABILITY) as fh:
        rel = json.load(fh)
    mulleder_retest = {
        row["amino_acid"]: {
            k: float(row[k]) for k in row if isinstance(row[k], (int, float)) and k != "n"
        }
        for row in rel["per_amino_acid"]
    }
    # Cooper's replicate strains: the loader's duplicate ledger against the served row.
    with open(osp.join(COOPER, "preprocess", "dropped_records.json")) as fh:
        ledger = json.load(fh)["duplicate_strain_rows"]
    served = coo.reset_index().set_index("row_index")
    cooper_retest: list[Retest] = []
    for key in COOPER_KEYS:
        a, b = [], []
        for row in ledger:
            if key not in row["levels"]:
                continue
            kept = served.loc[row["kept_row_index"], key]
            if np.isfinite(kept):
                a.append(float(row["levels"][key]))
                b.append(float(kept))
        if len(a) >= 10:
            x, y = np.array(a), np.array(b)
            pos = (x > 0) & (y > 0)
            cooper_retest.append(
                Retest(
                    key=key,
                    n_pairs=len(a),
                    spearman=float(spearmanr(x, y).correlation),
                    pearson_log=float(pearsonr(np.log10(x[pos]), np.log10(y[pos]))[0]),
                )
            )

    # ---- figure -----------------------------------------------------------------------
    plt.rcParams.update(
        {"font.family": "Arial", "font.size": 6, "svg.fonttype": "none",
         "axes.linewidth": 0.5}
    )
    # Layout in millimeters, from the bottom: panel b's labels, panel b, the legend
    # strip, panel a's labels, panel a; the whole stays under the 170 mm page height.
    cell_mm = 5.4
    heat_w, heat_h = cell_mm * len(MULLEDER_KEYS), cell_mm * len(COOPER_KEYS)
    fig_w = PANEL_WIDTHS_MM["full"]
    b_labels, bar_h, legend_gap, a_labels, top = 24.0, 27.0, 7.0, 21.0, 3.0
    fig_h = b_labels + bar_h + legend_gap + a_labels + heat_h + top
    fig = plt.figure(figsize=(mm_to_in(fig_w), mm_to_in(fig_h)))
    left = 34.0 / fig_w
    a_bottom = (b_labels + bar_h + legend_gap + a_labels) / fig_h
    ax = fig.add_axes([left, a_bottom, heat_w / fig_w, heat_h / fig_h])
    cmap = LinearSegmentedColormap.from_list(
        "corr", [PLOT_PALETTE[1], "#FFFFFF", PLOT_PALETTE[2]]
    )
    vmax = float(np.nanmax(np.abs(matrix)))
    im = ax.imshow(
        matrix, cmap=cmap, norm=TwoSlopeNorm(0.0, -vmax, vmax), aspect="equal",
        interpolation="nearest",
    )
    for i in range(len(COOPER_KEYS)):
        for j in range(len(MULLEDER_KEYS)):
            if np.isfinite(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=6)
    for peak, names in PEAK_MAP.items():
        for n in names:
            ax.add_patch(
                Rectangle(
                    (MULLEDER_KEYS.index(n) - 0.5, COOPER_KEYS.index(peak) - 0.5), 1, 1,
                    fill=False, edgecolor="black", linewidth=0.9,
                )
            )
    ax.set_xticks(range(len(MULLEDER_KEYS)))
    ax.set_xticklabels(MULLEDER_KEYS, rotation=90)
    ax.set_yticks(range(len(COOPER_KEYS)))
    ax.set_yticklabels([k.replace("+", " + ").replace("_", " ") for k in COOPER_KEYS])
    ax.set_xlabel("Mulleder 2016 amino acid (mM, minimal medium, prototroph)")
    ax.set_ylabel("Cooper 2010 peak (ratio to plate mean, complete medium)")
    ax.tick_params(length=2, width=0.5)
    for s in ax.spines.values():
        s.set_linewidth(0.5)
    cax = fig.add_axes(
        [left + (heat_w + 3.0) / fig_w, a_bottom, 3.0 / fig_w, heat_h / fig_h]
    )
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Spearman over shared strains")
    cb.outline.set_linewidth(0.5)
    ax.text(-0.16, 1.01, "a", transform=ax.transAxes, fontsize=8, fontweight="bold",
            ha="left", va="bottom")

    # b: the matched pairs beside both retest references.
    ax2 = fig.add_axes([left, b_labels / fig_h, (fig_w - 34.0 - 8.0) / fig_w, bar_h / fig_h])
    labels = [f"{short(s.cooper_peak)} : {short(s.mulleder)}" for s in matched]
    xs = np.arange(len(matched))
    ax2.bar(xs, [s.spearman for s in matched], color=PLOT_PALETTE[0], edgecolor="black",
            linewidth=0.5, width=0.7, label="cross-screen Spearman")
    mul_ref = [
        np.mean([mulleder_retest[n]["spearman_retest_log10"] for n in s.mulleder.split("+")])
        for s in matched
    ]
    coo_ref = {r.key: r.spearman for r in cooper_retest}
    ax2.plot(xs, mul_ref, linestyle="none", marker="o", markersize=3,
             markerfacecolor=PLOT_PALETTE[2], markeredgecolor="black", markeredgewidth=0.4,
             label="Mulleder retest Spearman (194 strains)")
    ax2.plot(xs, [coo_ref.get(s.cooper_peak, np.nan) for s in matched], linestyle="none",
             marker="s", markersize=3, markerfacecolor=PLOT_PALETTE[3],
             markeredgecolor="black", markeredgewidth=0.4,
             label="Cooper replicate Spearman (duplicate strains)")
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.set_xticks(xs)
    ax2.set_xticklabels(labels, rotation=90)
    ax2.set_ylabel("Spearman")
    ax2.set_ylim(-0.2, 1.0)
    ax2.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.2))
    ax2.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.1))
    ax2.grid(axis="y", which="both", linewidth=0.3, color="#CCCCCC")
    ax2.tick_params(which="minor", length=0)
    ax2.tick_params(length=2, width=0.5)
    ax2.legend(loc="lower left", bbox_to_anchor=(0, 1.03), ncol=3, frameon=False,
               fontsize=6, handlelength=1.2, columnspacing=1.0)
    ax2.text(-0.16 * heat_w / (fig_w - 42.0), 1.03, "b", transform=ax2.transAxes,
             fontsize=8, fontweight="bold", ha="right", va="bottom")

    os.makedirs(IMAGES, exist_ok=True)
    stamp = timestamp()
    svg = osp.join(IMAGES, f"amino_acid_dataset_concordance_{stamp}.svg")
    savefig_true_size_svg(fig, svg)
    fig.savefig(svg.replace(".svg", ".png"), dpi=300)
    plt.close(fig)

    report = Report(
        mulleder_manifest=manifest(MULLEDER),
        cooper_manifest=manifest(COOPER),
        n_mulleder=int(len(mul)),
        n_cooper=int(len(coo)),
        n_shared_strains=len(shared),
        peak_map=PEAK_MAP,
        matched=matched,
        matrix_spearman=[[None if not np.isfinite(v) else float(v) for v in row] for row in matrix],
        matrix_n=counts.tolist(),
        cooper_keys=COOPER_KEYS,
        mulleder_keys=MULLEDER_KEYS,
        mulleder_retest=mulleder_retest,
        cooper_duplicate_retest=cooper_retest,
        figure=svg,
        note=(
            "Spearman over the strains both screens measured; Pearson on log10 of both "
            "quantities over strains with positive values in both; top-5-percent overlap "
            "with the hypergeometric tail probability. Cooper composite peaks are matched "
            "to the sum of their components' mM and to each component."
        ),
    )
    os.makedirs(RESULTS, exist_ok=True)
    out = osp.join(RESULTS, "amino_acid_dataset_concordance.json")
    with open(out, "w") as fh:
        fh.write(report.model_dump_json(indent=2))
    print(f"Cooper replicate retest: {[(r.key, round(r.spearman, 3), r.n_pairs) for r in cooper_retest]}")
    print(svg)
    print(out)


if __name__ == "__main__":
    main()
