# experiments/019-simb-multimodal/scripts/plot_knn_embedding_probe.py
# [[experiments.019-simb-multimodal.scripts.plot_knn_embedding_probe]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/plot_knn_embedding_probe
"""Publication figure for the kNN embedding probe.

Reads results/knn_embedding_probe.json (produced by knn_embedding_probe.py) and renders a
two-panel figure -- morphology and expression -- of the best-k per-feature Pearson for each
embedding, grouped by representational axis.

The figure has to carry three comparisons at once, so all three are drawn as reference
lines rather than left to the caption:

  * NOISE FLOOR (shaded band): the largest |r| reached by ANY random control. Random
    vectors carry no signal by construction, so whatever they score is the metric's noise
    floor at this validation size -- and it is NOT zero: random_1000 reaches ~0.036 simply
    because a 1000-d random geometry picks arbitrary neighbours and n_val is finite. An
    embedding must clear this band to mean anything, which is exactly the calibration the
    random ladder (1 / 10 / 100 / 1000 dims) was added to provide.
  * TRANSFORMER (dashed line): the best the swept CGT achieved on the same split and the
    same metric. Anything above it is beaten by a parameter-free similarity average.
  * one_hot_gene is drawn as a hatched "undefined" bar: one-hot vectors are mutually
    orthogonal, so every cosine similarity is 0 and kNN has no notion of a similar gene.

Palette/format follow the repo standard (CLAUDE.md "Figure & Plotting Standards"):
torchcell.utils.PLOT_PALETTE, strict panel width from PANEL_WIDTHS_MM, Arial 6 pt, all
four spines, tenth gridlines, true-size SVG export, PNG written alongside as a raster
fallback.

Run from repo root:
    python experiments/019-simb-multimodal/scripts/plot_knn_embedding_probe.py
"""

from __future__ import annotations

import json
import os
import os.path as osp

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from matplotlib.ticker import MultipleLocator

from torchcell.utils import PANEL_WIDTHS_MM, PLOT_PALETTE, mm_to_in, savefig_true_size_svg

load_dotenv("/home/michaelvolk/Documents/projects/torchcell/.env")

# Best swept-CGT per-feature Pearson on this split, for the reference line.
TRANSFORMER = {"morphology": 0.0333, "expression": 0.0795}

# (display label, results key, axis) -- axis drives the colour, ordered so the warm
# primaries land on the axes that carry signal.
AXES_ORDER = [
    ("prot_T5", "prot_T5_all", "Protein LM"),
    ("prot_T5 (no dub.)", "prot_T5_no_dubious", "Protein LM"),
    ("ESM2 650M", "esm2_t33_650M_UR50D_all", "Protein LM"),
    ("ESM2 (no dub.)", "esm2_t33_650M_UR50D_no_dubious", "Protein LM"),
    ("chrom. pathways", "normalized_chrom_pathways", "Pathway / graph"),
    ("CaLM", "calm", "Codon / ORF"),
    ("codon freq.", "codon_frequency", "Codon / ORF"),
    ("species LM 5'", "species_lm_five_prime", "Regulatory DNA"),
    ("species LM 3'", "species_lm_three_prime", "Regulatory DNA"),
    ("species LM 5'+3'", "species_lm_5p_3p", "Regulatory DNA"),
    ("NT locus 5979", "nt_window_5979", "Nucleotide LM"),
    ("NT 5' 1003", "nt_window_five_prime_1003", "Nucleotide LM"),
    ("NT 3' 300", "nt_window_three_prime_300", "Nucleotide LM"),
    ("NT 5'+3'", "nt_5prime_3prime", "Nucleotide LM"),
    ("one-hot gene", "one_hot_gene", "Identity only"),
    ("random d=1", "random_1", "Random control"),
    ("random d=10", "random_10", "Random control"),
    ("random d=100", "random_100", "Random control"),
    ("random d=1000", "random_1000", "Random control"),
]
AXIS_COLOR = {
    "Protein LM": PLOT_PALETTE[0],
    "Pathway / graph": PLOT_PALETTE[1],
    "Codon / ORF": PLOT_PALETTE[2],
    "Regulatory DNA": PLOT_PALETTE[3],
    "Nucleotide LM": PLOT_PALETTE[4],
    "Identity only": PLOT_PALETTE[5],
    "Random control": PLOT_PALETTE[5],
}
RANDOM_KEYS = ["random_1", "random_10", "random_100", "random_1000"]


def _style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 6,
            "axes.labelsize": 6,
            "axes.titlesize": 6,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
            "axes.linewidth": 0.5,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "svg.fonttype": "none",
        }
    )


def main() -> None:
    _style()
    # Resolve results relative to THIS script, not $EXPERIMENT_ROOT: the env var points at
    # the primary checkout, so a worktree run would read the wrong tree's results.
    results_dir = osp.abspath(osp.join(osp.dirname(__file__), "..", "results"))
    images_dir = osp.join(os.environ["ASSET_IMAGES_DIR"], "019-simb-multimodal")
    os.makedirs(images_dir, exist_ok=True)

    with open(osp.join(results_dir, "knn_embedding_probe.json")) as f:
        res = json.load(f)
    arms = res["arms"]

    modalities = ["morphology", "expression"]
    # Noise floor = the largest |r| any random control reaches, per modality.
    floor = {
        m: max(
            abs(arms[k]["modalities"][m]["best_pearson_per_feature"])
            for k in RANDOM_KEYS
            if k in arms
        )
        for m in modalities
    }

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(78.0)),
        sharey=True,
    )

    for ax, mod in zip(axes, modalities):
        vals, colors, labels, undefined = [], [], [], []
        for label, key, axis_name in AXES_ORDER:
            arm = arms.get(key)
            if arm is None:
                continue
            m = arm["modalities"][mod]
            deg = bool(m.get("degenerate_orthogonal_geometry", False)) or not np.isfinite(
                m["best_pearson_per_feature"]
            )
            labels.append(f"{label}  ({arm['dim']}d)")
            vals.append(0.0 if deg else m["best_pearson_per_feature"])
            colors.append(AXIS_COLOR[axis_name])
            undefined.append(deg)

        y = np.arange(len(vals))
        ax.axhspan(-0.5, len(vals) - 0.5, xmin=0, xmax=0, color="none")  # keep limits sane
        # Noise-floor band, drawn first so bars sit on top of it.
        ax.axvspan(
            -floor[mod],
            floor[mod],
            color=PLOT_PALETTE[5],
            alpha=0.12,
            lw=0,
            zorder=0,
            label=f"noise floor (|r| < {floor[mod]:.3f})",
        )
        bars = ax.barh(
            y, vals, color=colors, edgecolor="black", linewidth=0.4, height=0.72, zorder=2
        )
        for b, deg in zip(bars, undefined):
            if deg:
                b.set_hatch("///")
                b.set_facecolor("white")
        ax.axvline(
            TRANSFORMER[mod],
            color="black",
            lw=0.8,
            ls="--",
            zorder=3,
            label=f"swept transformer ({TRANSFORMER[mod]:.3f})",
        )
        ax.axvline(0.0, color="black", lw=0.5, zorder=1)

        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.set_xlabel("per-feature Pearson $r$ (best $k$)")
        ax.set_title(
            f"{mod}  (n$_{{val}}$="
            f"{arms['prot_T5_all']['modalities'][mod]['n_val']} genes)",
            pad=3,
        )
        ax.xaxis.set_major_locator(MultipleLocator(0.05))
        ax.xaxis.set_minor_locator(MultipleLocator(0.025))
        ax.tick_params(which="minor", length=0)
        ax.grid(axis="x", which="both", lw=0.3, color="0.85", zorder=0)
        ax.set_axisbelow(True)
        for s in ax.spines.values():
            s.set_visible(True)
        ax.legend(loc="lower right", frameon=True, framealpha=0.95, borderpad=0.3)

    axes[0].invert_yaxis()  # once only -- the y axis is shared between panels

    # Annotate the undefined bar once, on the left panel, anchored to its real row.
    one_hot_row = next(
        i for i, (_, k, _) in enumerate(AXES_ORDER) if k == "one_hot_gene"
    )
    axes[0].annotate(
        "undefined: one-hot vectors are mutually\northogonal, so every cosine similarity is 0\nand kNN has no notion of a similar gene",
        xy=(0.002, one_hot_row),
        xytext=(0.042, one_hot_row - 3.2),
        fontsize=5,
        arrowprops=dict(arrowstyle="->", lw=0.4, color="black"),
    )

    fig.tight_layout(pad=0.6)
    stem = osp.join(images_dir, "knn_embedding_probe")
    fig.savefig(f"{stem}.png", dpi=300)
    savefig_true_size_svg(fig, f"{stem}.svg")
    print(f"wrote {stem}.png")
    print(f"wrote {stem}.svg")
    print(f"noise floor: {floor}")


if __name__ == "__main__":
    main()
