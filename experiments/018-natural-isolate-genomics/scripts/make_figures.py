# experiments/018-natural-isolate-genomics/scripts/make_figures.py
# [[experiments.018-natural-isolate-genomics.scripts.make_figures]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/018-natural-isolate-genomics/scripts/make_figures

"""Figures for issue #66 -- natural-isolate genomic diversity vs KO-expression variability.

Follows the repo-wide **Figure & Plotting Standards** in CLAUDE.md:

* Colour comes from the single ordered source of truth, ``torchcell.utils.PLOT_PALETTE``,
  assigned IN ORDER. This is a 2-series comparison, so it takes the first two slots --
  the warm primaries orange ``#D79B00`` and red ``#B85450``. Blue/gray are deliberately
  NOT used (the rule: they are not reached until the four primaries are used up), which
  is why an earlier draft of these figures -- blue-first, off-palette -- was wrong.
* Colour is assigned by ENTITY and held fixed across every panel, never by rank::

      engineered KO (Kemmeren / Sameith)   -> PLOT_PALETTE[0]  orange
      natural isolate (Caudal / Peter)     -> PLOT_PALETTE[1]  red

  The same two do double duty in the region panel as coding vs non-coding, which is a
  real identity distinction rather than decoration.
* Panel width is STRICT (``PANEL_WIDTHS_MM``), height loose and <= ``MAX_HEIGHT_MM``.
* Boxed axes (all four spines), Arial 6 pt, black bar edges at full opacity.
* Exported as a true-size SVG (``savefig_true_size_svg``) for draw.io/paper, plus a
  high-DPI PNG for the dendron note.

Validator check on the pair actually used (light surface): lightness band PASS, chroma
floor PASS, CVD separation DeltaE 47.9 (deutan) -- far above the >= 12 target. Orange sits
at 2.38:1 contrast on white, below 3:1, so the **relief rule** applies: every bar carries
a direct value label and identity never rests on colour alone.

DELIBERATE DEVIATION (stated, per the standard): these are multi-panel analytic figures
read on screen from a Dendron note, not final print panels. They are authored at the
``full`` (179 mm) width with 6 pt type as the standard requires, but the PNG is rendered
at 300 dpi so the 6 pt text is legible on screen.
"""

import json
import os
import os.path as osp

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from matplotlib.colors import to_rgba

from torchcell.timestamp import timestamp
from torchcell.utils import (
    MAX_HEIGHT_MM,
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    PLOT_PALETTE_FILL,
    mm_to_in,
    savefig_true_size_svg,
)

load_dotenv()

DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
ASSET_IMAGES_DIR = os.environ["ASSET_IMAGES_DIR"]

EXP_DIR = osp.join(EXPERIMENT_ROOT, "018-natural-isolate-genomics")
RESULTS_DIR = osp.join(EXP_DIR, "results")
IMG_DIR = osp.join(ASSET_IMAGES_DIR, "018-natural-isolate-genomics")

# Repo standard: Arial 6 pt everywhere; real (not path) text in the SVG.
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 6,
        "axes.titlesize": 6,
        "axes.labelsize": 6,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "figure.titlesize": 7,
        "svg.fonttype": "none",
        "axes.linewidth": 0.5,
        "lines.linewidth": 1.0,
        "patch.linewidth": 0.5,
    }
)

# Entity colours -- PLOT_PALETTE assigned IN ORDER (see module docstring).
KO = PLOT_PALETTE[0]  # orange -- engineered knockout / coding
NAT_FILL = PLOT_PALETTE_FILL[1]  # light red -- the lighter member of a 2-level bar
NAT = PLOT_PALETTE[1]  # red    -- natural isolate / non-coding
INK = "#000000"
INK2 = "#4A4A4A"

FULL_IN = mm_to_in(PANEL_WIDTHS_MM["full"])  # 179 mm -> 7.05 in
MAX_H_IN = mm_to_in(MAX_HEIGHT_MM)  # 170 mm cap

# Figures have converged, so filenames are STABLE (no timestamp) -- regenerating
# overwrites in place and the note's image references never go stale. Set
# TIMESTAMP_FIGURES=1 to go back to timestamped output while iterating.
TS = timestamp() if os.environ.get("TIMESTAMP_FIGURES") else None


CAPTIONS: dict[str, str] = {}


def _save(fig, name: str, caption: str | None = None) -> str:
    """Write the panel; the caption goes to the NOTE, never baked into the canvas.

    The standard forbids ``bbox_inches="tight"`` on a fixed-width panel (it recrops and
    defeats the width template), so a ``fig.text`` caption hung below the axes simply
    falls off the canvas. Captions belong in the note/paper anyway -- we collect them
    here and emit them as a manifest the note quotes.
    """
    os.makedirs(IMG_DIR, exist_ok=True)
    if caption:
        CAPTIONS[name] = " ".join(caption.split())
    stem = f"{name}_{TS}" if TS else name
    png = osp.join(IMG_DIR, f"{stem}.png")
    svg = osp.join(IMG_DIR, f"{stem}.svg")
    # PNG for the note (300 dpi keeps 6 pt legible on screen); true-size SVG for draw.io.
    fig.savefig(png, dpi=300, facecolor="white")
    savefig_true_size_svg(fig, svg, facecolor="white")
    plt.close(fig)
    print(f"      -> {png}", flush=True)
    return png


def _despine(ax):
    """Repo standard: BOX the plot -- all four spines, 0.5 pt black."""
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(True)
        ax.spines[s].set_color(INK)
        ax.spines[s].set_linewidth(0.5)
    ax.tick_params(colors=INK, width=0.5, length=2)
    ax.grid(True, alpha=0.15, linewidth=0.4, color=INK2)
    ax.set_axisbelow(True)


def _headroom(ax, frac: float = 0.30) -> None:
    """Extend the top of the y-axis so value labels and legends never overlap marks."""
    lo, hi = ax.get_ylim()
    if ax.get_yscale() == "log":
        ax.set_ylim(lo, hi * (10**frac))
    else:
        ax.set_ylim(lo, hi + (hi - lo) * frac)


def fig_genome_divergence():
    print("[fig] genome divergence", flush=True)
    pg = pd.read_parquet(osp.join(RESULTS_DIR, "per_gene_divergence_summary.parquet"))
    ps = pd.read_parquet(osp.join(RESULTS_DIR, "per_strain_divergence_summary.parquet"))

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(FULL_IN, min(FULL_IN * 0.3833, MAX_H_IN)),
        constrained_layout=True,
    )

    ax = axes[0]
    v = pg["total_divergence_mean"].dropna() * 100
    ax.hist(v, bins=80, color=NAT, edgecolor=INK, linewidth=0.25)
    _headroom(ax)
    med = float(v.median())
    ax.axvline(med, color=INK, linestyle="--", linewidth=1.6)
    ax.annotate(
        f"median {med:.2f}%",
        xy=(med, ax.get_ylim()[1] * 0.88),
        xytext=(8, 0),
        textcoords="offset points",
        color=INK,
        fontsize=5.5,
    )
    ax.set_xlabel("mean divergence from S288C across isolates (%)")
    ax.set_ylabel("reference ORFs")
    ax.set_title("Per-ORF divergence (6,011 ORFs)", loc="left", color=INK)
    ax.set_xlim(0, min(float(v.quantile(0.995)), 5))
    _despine(ax)

    ax = axes[1]
    w = ps["genome_wide_divergence"].dropna() * 100
    ax.hist(w, bins=60, color=NAT, edgecolor=INK, linewidth=0.25)
    _headroom(ax)
    med = float(w.median())
    ax.axvline(med, color=INK, linestyle="--", linewidth=1.6)
    ax.annotate(
        f"median {med:.2f}%",
        xy=(med, ax.get_ylim()[1] * 0.88),
        xytext=(8, 0),
        textcoords="offset points",
        color=INK,
        fontsize=5.5,
    )
    ax.set_xlabel("genome-wide SNP divergence from S288C (%)")
    ax.set_ylabel("isolates")
    ax.set_title("Per-isolate divergence (1,011 isolates)", loc="left", color=INK)
    _despine(ax)

    fig.suptitle(
        "Natural isolates differ from S288C by well under 1% of coding bases",
        x=0.005,
        ha="left",
        color=INK,
    )
    return _save(
        fig,
        "genome_divergence",
        "SNP divergence is het-weighted per Peter 2018's published convention; "
        "length-changing indel alleles are scored by exact edit distance.",
    )


def fig_pangenome():
    print("[fig] pangenome + natural KO burden", flush=True)
    po = pd.read_parquet(osp.join(RESULTS_DIR, "pangenome_orf_presence.parquet"))
    bd = pd.read_parquet(osp.join(RESULTS_DIR, "natural_ko_burden.parquet"))

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(FULL_IN, min(FULL_IN * 0.3833, MAX_H_IN)),
        constrained_layout=True,
    )

    ax = axes[0]
    ax.hist(
        po["frac_isolates_present"] * 100,
        bins=50,
        color=NAT,
        edgecolor=INK,
        linewidth=0.25,
    )
    ax.set_yscale("log")
    _headroom(ax, 0.55)
    n_core = int(po["is_core_peter"].sum())
    n_var = int((~po["is_core_peter"]).sum())
    ax.annotate(
        f"core\n{n_core:,} ORFs",
        xy=(99, n_core),
        xytext=(-8, 12),
        textcoords="offset points",
        ha="right",
        va="bottom",
        color=INK,
        fontsize=5.5,
    )
    ax.annotate(
        f"variable: {n_var:,} ORFs", xy=(50, 30), color=INK2, fontsize=5.5, ha="center"
    )
    ax.set_xlabel("% of the 1,011 isolates carrying the ORF")
    ax.set_ylabel("pangenome ORFs (log)")
    ax.set_title("Pangenome presence spectrum", loc="left", color=INK)
    _despine(ax)

    # Only gene ABSENCE is a genuine null. Frameshift / premature stop are PREDICTED
    # loss-of-function in a co-evolved background -- never verified as nulls -- so they
    # are shown as a separate, clearly-labelled series rather than folded into one
    # "broken" count. (The earlier "broken ORFs" framing was an overclaim.)
    ax = axes[1]
    ax.hist(
        bd["n_absent"],
        bins=50,
        color=NAT,
        edgecolor=INK,
        linewidth=0.25,
        label="gene ABSENT (a genuine null)",
    )
    ax.hist(
        bd["n_frameshift"] + bd["n_premature_stop"],
        bins=50,
        # a lighter RED (translucent NAT), not the pinkish PLOT_PALETTE_FILL[1] --
        # the black edge stays opaque since edgecolor is set separately.
        color=to_rgba(NAT, 0.5),
        edgecolor=INK,
        linewidth=0.25,
        label="frameshift / premature stop (PREDICTED LoF, unverified)",
    )
    ax.axvline(
        1, color=KO, linewidth=2.6, label="Kemmeren engineered KO = 1 (verified null)"
    )
    _headroom(ax, 0.40)
    med = float(bd["n_absent"].median())
    ax.axvline(med, color=INK, linestyle="--", linewidth=1.6)
    ax.annotate(
        f"median {med:.0f}\nabsent",
        xy=(med, ax.get_ylim()[1] * 0.55),
        xytext=(10, 0),
        textcoords="offset points",
        color=INK,
        fontsize=5.5,
    )
    ax.set_xlabel("reference ORFs per isolate")
    ax.set_ylabel("isolates")
    ax.set_title("Natural gene loss vs one engineered KO", loc="left", color=INK)
    ax.legend(frameon=False, loc="upper right", fontsize=5)
    _despine(ax)

    fig.suptitle(
        "A natural isolate is missing a median of 123 reference ORFs outright",
        x=0.005,
        ha="left",
        color=INK,
    )
    return _save(
        fig,
        "pangenome_and_ko_burden",
        "Core = present in all 1,011 isolates (Peter 2018's own definition; we recover "
        "4,942 vs their published 4,940).\nNOT comparable to a KO one-for-one: a KanMX "
        "deletion is a verified complete null in an isogenic background; a natural "
        "variant allele is unverified, selected-upon and compensated.",
    )


def fig_regional_diversity():
    print("[fig] regional nucleotide diversity", flush=True)
    df = pd.read_parquet(
        osp.join(RESULTS_DIR, "regulatory_divergence_by_region.parquet")
    )
    with open(osp.join(RESULTS_DIR, "regulatory_divergence_summary.json")) as fh:
        s = json.load(fh)

    order = ["cds", "upstream_1000", "downstream_297", "intergenic_other"]
    pretty = {
        "cds": "CDS\n(coding)",
        "upstream_1000": "1,000 bp\nupstream",
        "downstream_297": "297 bp\ndownstream",
        "intergenic_other": "other\nintergenic",
    }
    df = df.set_index("region").loc[order].reset_index()
    # coding vs non-coding is a real identity split, not decoration
    colors = [KO] + [NAT] * 3

    fig, ax = plt.subplots(
        figsize=(FULL_IN, min(FULL_IN * 0.5333, MAX_H_IN)), constrained_layout=True
    )
    bars = ax.bar(
        [pretty[r] for r in df.region],
        df.pi_percent,
        color=colors,
        edgecolor=INK,
        linewidth=0.5,
        width=0.66,
    )
    ax.set_ylim(0, float(df.pi_percent.max()) * 1.32)
    # direct value labels on every bar (relief rule: identity never colour-alone)
    for b, v in zip(bars, df.pi_percent):
        ax.annotate(
            f"{v:.3f}%",
            xy=(b.get_x() + b.get_width() / 2, v),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            color=INK,
            fontsize=6,
        )
    ax.set_ylabel("nucleotide diversity  π  (%)")
    ax.set_title(
        "Regulatory sequence is ~2× more polymorphic than coding sequence",
        loc="left",
        color=INK,
        fontsize=6.5,
    )
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=KO),
        plt.Rectangle((0, 0), 1, 1, color=NAT),
    ]
    ax.legend(
        handles, ["coding", "non-coding"], frameon=False, loc="upper left", ncols=2
    )
    _despine(ax)
    saw = s["species_aware_transformer_window"]
    return _save(
        fig,
        "regional_nucleotide_diversity",
        f"The species-aware transformer's input (CDS + 1,000 bp upstream + 297 bp "
        f"downstream) covers {100 * saw['frac_of_genome_covered']:.1f}% of the genome "
        f"and captures {100 * saw['frac_of_pi_captured']:.1f}% of all nucleotide "
        f"diversity.\nπ from Peter 2018's 1011Matrix.gvcf (1,753,947 variants); regions "
        f"assigned by precedence CDS > upstream > downstream > intergenic.",
    )


def fig_de_comparison():
    print("[fig] DE comparison (headline)", flush=True)
    de = pd.read_parquet(osp.join(RESULTS_DIR, "de_counts_per_strain.parquet"))
    with open(osp.join(RESULTS_DIR, "de_comparison_summary.json")) as fh:
        s = json.load(fh)

    kem = de[de.dataset == "kemmeren2014_single_ko"]["n_de_paper_exact"].dropna()
    cau = de[de.dataset == "caudal2024_natural_isolate"]["n_de_paper_exact"].dropna()

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(FULL_IN, min(FULL_IN * 0.3840, MAX_H_IN)),
        constrained_layout=True,
    )

    ax = axes[0]
    bins = np.logspace(0, np.log10(max(kem.max(), cau.max()) + 1), 45)
    ax.hist(
        np.clip(kem, 1, None),
        bins=bins,
        color=to_rgba(KO, 0.85),  # face only; edge stays solid black (see standard)
        edgecolor=INK,
        linewidth=0.25,
        label="Kemmeren single KO (n=1,484)",
    )
    ax.hist(
        np.clip(cau, 1, None),
        bins=bins,
        color=to_rgba(NAT, 0.72),  # face only; edge stays solid black
        edgecolor=INK,
        linewidth=0.25,
        label="Caudal natural isolate (n=943)",
    )
    ax.set_xscale("log")
    _headroom(ax, 0.34)
    km, cm = float(kem.median()), float(cau.median())
    ax.axvline(km, color=KO, linestyle="--", linewidth=2)
    ax.axvline(cm, color=NAT, linestyle="--", linewidth=2)
    # medians sit at 55% height, clear of the legend box in the upper right
    ax.annotate(
        f"median {km:.0f}",
        xy=(km, ax.get_ylim()[1] * 0.55),
        xytext=(-6, 0),
        textcoords="offset points",
        ha="right",
        color=KO,
        fontsize=5.5,
    )
    ax.annotate(
        f"median {cm:.0f}",
        xy=(cm, ax.get_ylim()[1] * 0.55),
        xytext=(8, 0),
        textcoords="offset points",
        color=NAT,
        fontsize=5.5,
    )
    ax.set_xlabel("differentially expressed genes per strain  (log)")
    ax.set_ylabel("strains")
    ax.set_title("DE genes per strain", loc="left", color=INK)
    ax.legend(frameon=False, loc="upper right")
    _despine(ax)

    ax = axes[1]
    # Use the matrices differential_expression_comparison.py writes -- the deleteome M
    # values it actually calls DE on. (kemmeren_M_matrix is genes x mutants.)
    kl = np.load(osp.join(RESULTS_DIR, "kemmeren_M_matrix.npy"))
    cl = np.load(osp.join(RESULTS_DIR, "caudal_log2_matrix.npy"))
    kf = kl[np.isfinite(kl)]
    cf = cl[np.isfinite(cl)]
    bins = np.linspace(-3, 3, 160)
    ax.hist(
        cf,
        bins=bins,
        density=True,
        color=to_rgba(NAT, 0.55),
        histtype="stepfilled",
        edgecolor="none",
        label="Caudal natural isolate",
    )
    ax.hist(
        kf,
        bins=bins,
        density=True,
        color=to_rgba(KO, 0.72),
        histtype="stepfilled",
        edgecolor="none",
        label="Kemmeren single KO",
    )
    t = float(np.log2(1.7))
    for x in (-t, t):
        ax.axvline(x, color=INK, linestyle=":", linewidth=1.3)
    ax.set_yscale("log")
    _headroom(ax, 0.55)
    ax.annotate(
        "FC = 1.7",
        xy=(t, ax.get_ylim()[0] * 3.2),
        xytext=(6, 0),
        textcoords="offset points",
        color=INK2,
        fontsize=5,
    )
    ax.set_xlabel("log2 expression ratio vs reference")
    ax.set_ylabel("density (log)")
    ax.set_title(
        f"Spread of the response:  SD {kf.std():.2f} (KO)  vs  {cf.std():.2f} (isolate)",
        loc="left",
        color=INK,
    )
    ax.legend(frameon=False, loc="upper right")
    _despine(ax)

    h = s["headline"]
    fig.suptitle(
        f"A natural isolate differentially expresses "
        f"{h['fold_more_genes_median']:.0f}× more genes than a single knockout "
        f"(median {h['caudal_natural_isolate_median_de']:.0f} vs "
        f"{h['kemmeren_single_ko_median_de']:.0f})",
        x=0.005,
        ha="left",
        color=INK,
    )
    return _save(
        fig,
        "de_comparison_ko_vs_natural",
        "Identical rule on both arms: |log2 FC| > log2(1.7) AND BH-adjusted p < 0.05 "
        "(Kemmeren 2014's own criterion). Caudal has no published p-values, so its "
        "noise model is estimated from its 29 replicate cultures\nand applied the same "
        "way — without that control its count would be 1,011, not 160.",
    )


def fig_bit_ledger():
    p = osp.join(RESULTS_DIR, "bit_ledger.parquet")
    if not osp.exists(p):
        print(
            "[fig] bit ledger -- SKIPPED (bit_accounting.py has not finished)",
            flush=True,
        )
        return None
    print("[fig] bit ledger", flush=True)
    df = pd.read_parquet(p)

    mods = [
        ("kemmeren2014_single_ko", "Kemmeren\nsingle KO", KO),
        ("sameith2015_double_ko", "Sameith\ndouble KO", KO),
        ("caudal2024_natural_isolate", "Caudal\nnatural isolate", NAT),
    ]
    rows = []
    for m, label, c in mods:
        d = df[df.modality == m]

        def get(enc):
            r = d[d.encoding == enc]
            return float(r["bits_per_strain"].iloc[0]) if len(r) else np.nan

        gt = get("genotype_minimal")
        if not np.isfinite(gt):
            gt = get("sequence_diff_vs_S288C")
        rows.append(
            {
                "label": label,
                "color": c,
                "genotype_bits": gt,
                "genotype_as_stored": get("genotype_as_stored"),
                "phenotype_bits": get("phenotype_values_only"),
                "phenotype_as_stored": get("phenotype_as_stored"),
            }
        )
    r = pd.DataFrame(rows)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(FULL_IN, min(FULL_IN * 0.3840, MAX_H_IN)),
        constrained_layout=True,
    )

    ax = axes[0]
    x = np.arange(len(r))
    w = 0.36
    b1 = ax.bar(
        x - w / 2,
        r.genotype_bits,
        w,
        color=KO,
        edgecolor=INK,
        linewidth=0.5,
        label="genotype (minimal encoding)",
    )
    b2 = ax.bar(
        x + w / 2,
        r.phenotype_bits,
        w,
        color=NAT,
        edgecolor=INK,
        linewidth=0.5,
        label="phenotype (values only)",
    )
    ax.set_yscale("log")
    _headroom(ax, 0.9)
    for bars in (b1, b2):
        for b in bars:
            v = b.get_height()
            if np.isfinite(v) and v > 0:
                ax.annotate(
                    f"{v:,.0f}" if v >= 1000 else f"{v:.1f}",
                    xy=(b.get_x() + b.get_width() / 2, v),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center",
                    color=INK,
                    fontsize=5,
                )
    ax.set_xticks(x)
    ax.set_xticklabels(r.label)
    ax.set_ylabel("bits per strain  (log)")
    ax.set_title("Genotype vs phenotype codelength per strain", loc="left", color=INK)
    ax.legend(frameon=False, loc="upper left")
    _despine(ax)

    ax = axes[1]
    ratio = r.phenotype_bits / r.genotype_bits
    bars = ax.bar(
        r.label, ratio, color=list(r.color), edgecolor=INK, linewidth=0.5, width=0.6
    )
    ax.set_yscale("log")
    _headroom(ax, 0.55)
    for b, v in zip(bars, ratio):
        ax.annotate(
            f"{v:,.0f}×" if v >= 10 else f"{v:.2f}×",
            xy=(b.get_x() + b.get_width() / 2, v),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            color=INK,
            fontsize=6,
        )
    ax.axhline(1, color=INK2, linewidth=1.2, linestyle=":")
    ax.annotate(
        "parity",
        xy=(0.98, 1),
        xycoords=("axes fraction", "data"),
        xytext=(0, 4),
        textcoords="offset points",
        ha="right",
        color=INK2,
        fontsize=5,
    )
    ax.set_ylabel("phenotype bits ÷ genotype bits  (log)")
    ax.set_title("Ratio of the two codelengths", loc="left", color=INK)
    _despine(ax)

    fig.suptitle(
        "Phenotype codelength is ~180 kbit in all three; the genotype spans 14.7 bits "
        "to 3.3 Mbit",
        x=0.005,
        ha="left",
        color=INK,
    )
    return _save(
        fig,
        "bit_ledger",
        "L_C = gzip codelength (zlib level 6, streamed) — a computable UPPER BOUND on "
        "Kolmogorov complexity, not an entropy, and a loose one (gzip leaves 1.4–5× vs "
        "a large-window compressor).\nThe right-hand ratio is arithmetic on those two "
        "codelengths; any reading of it as 'what a model must infer' is interpretation, "
        "not measurement.",
    )


def fig_coupling():
    print("[fig] genome -> expression coupling", flush=True)
    ps = pd.read_parquet(osp.join(RESULTS_DIR, "per_strain_divergence_summary.parquet"))
    de = pd.read_parquet(osp.join(RESULTS_DIR, "de_counts_per_strain.parquet"))
    bd = pd.read_parquet(osp.join(RESULTS_DIR, "natural_ko_burden.parquet"))

    cau = de[de.dataset == "caudal2024_natural_isolate"][
        ["strain", "n_de_paper_exact"]
    ].copy()
    ps["strain"] = ps["strain"].astype(str)
    cau["strain"] = cau["strain"].astype(str)
    bd["strain"] = bd["strain"].astype(str)
    j = cau.merge(ps, on="strain", how="inner").merge(bd, on="strain", how="inner")
    j = j[np.isfinite(j["n_de_paper_exact"]) & np.isfinite(j["genome_wide_divergence"])]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(FULL_IN, min(FULL_IN * 0.3833, MAX_H_IN)),
        constrained_layout=True,
    )

    y = j["n_de_paper_exact"]
    panels = [
        (
            j["genome_wide_divergence"] * 100,
            "genome-wide SNP divergence from S288C (%)",
            "Sequence divergence vs DE",
        ),
        (
            j["n_broken_union"],
            "reference ORFs absent or predicted-LoF\n"
            "(absent | frameshift | premature stop)",
            "Natural gene loss vs DE",
        ),
    ]
    rs = []
    for ax, (x, xlab, title) in zip(axes, panels):
        ax.scatter(x, y, s=13, color=NAT, alpha=0.5, edgecolor="none")
        r = float(np.corrcoef(x, y)[0, 1])
        rs.append(r)
        m, b = np.polyfit(x, y, 1)  # degree-1 = ordinary least-squares linear fit
        xs = np.linspace(float(x.min()), float(x.max()), 50)
        ax.plot(xs, m * xs + b, color=INK, linewidth=1.0)
        _headroom(ax, 0.18)
        ax.annotate(
            f"Pearson r = {r:.2f}   (n = {len(j)})",
            xy=(0.03, 0.94),
            xycoords="axes fraction",
            va="top",
            color=INK,
            fontsize=5.5,
        )
        ax.set_xlabel(xlab)
        ax.set_ylabel("differentially expressed genes")
        ax.set_title(title, loc="left", color=INK)
        _despine(ax)

    fig.suptitle(
        f"Genotype magnitude is a weak predictor of transcriptome response "
        f"(r = {rs[0]:.2f}, {rs[1]:.2f})",
        x=0.005,
        ha="left",
        color=INK,
    )
    return _save(
        fig,
        "genotype_phenotype_coupling",
        "Each point is one of the 943 natural isolates. If genome bits translated "
        "directly into transcriptome bits, these would be tight lines.",
    )


def fig_codon_usage():
    print("[fig] codon usage", flush=True)
    cc = pd.read_parquet(osp.join(RESULTS_DIR, "codon_counts_per_strain.parquet"))
    ref = np.load(osp.join(RESULTS_DIR, "reference_codon_counts.npy"))

    cols = [f"codon_{i}" for i in range(64)]
    M = cc[cols].to_numpy(dtype=float)
    freq = M / M.sum(axis=1, keepdims=True)
    rf = ref / ref.sum()
    delta = (freq - rf[None, :]) * 1000  # per-mille deviation from S288C

    bases = "ACGT"
    names = [f"{bases[i // 16]}{bases[(i // 4) % 4]}{bases[i % 4]}" for i in range(64)]
    order = np.argsort(-np.abs(delta).mean(axis=0))[:20]

    fig, ax = plt.subplots(
        figsize=(FULL_IN, min(FULL_IN * 0.4000, MAX_H_IN)), constrained_layout=True
    )
    data = [delta[:, i] for i in order]
    bp = ax.boxplot(
        data,
        patch_artist=True,
        widths=0.6,
        showfliers=False,
        medianprops=dict(color=INK, linewidth=1.8),
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(NAT)
        patch.set_alpha(0.75)
        patch.set_edgecolor(INK)
        patch.set_linewidth(0.5)
    ax.axhline(0, color=INK, linewidth=1.4)
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels([names[i] for i in order], rotation=45, ha="right")
    ax.set_xlabel("codon (20 most variable)")
    ax.set_ylabel("deviation from S288C usage  (‰)")
    mx = float(np.abs(delta).max())
    ax.set_title(
        f"Codon usage is essentially invariant across 1,011 isolates "
        f"(largest single-isolate deviation {mx:.2f}‰)",
        loc="left",
        color=INK,
    )
    _despine(ax)
    return _save(
        fig,
        "codon_usage_deviation",
        "Per-isolate codon frequencies over intronless reference ORFs, minus S288C's. "
        "A 1‰ shift is one codon in a thousand.",
    )


def main() -> None:
    os.makedirs(IMG_DIR, exist_ok=True)
    made = []
    for f in (
        fig_genome_divergence,
        fig_pangenome,
        fig_regional_diversity,
        fig_de_comparison,
        fig_coupling,
        fig_codon_usage,
        fig_bit_ledger,
    ):
        p = f()
        if p:
            made.append(p)
    # Captions live in the note, not on the canvas -- emit them so the note can quote
    # them verbatim and they never drift from the figure that produced them.
    with open(osp.join(RESULTS_DIR, "figure_captions.json"), "w") as fh:
        json.dump(CAPTIONS, fh, indent=2)

    print(f"\n{len(made)} figures (png + true-size svg) -> {IMG_DIR}")
    for p in made:
        print(f"  {osp.basename(p)}")
    print(f"captions -> {RESULTS_DIR}/figure_captions.json")


if __name__ == "__main__":
    main()
