# experiments/040-inhibitor-synergy-wetlab/scripts/lian_validation_feasibility.py
# [[experiments.040-inhibitor-synergy-wetlab.scripts.lian_validation_feasibility]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/040-inhibitor-synergy-wetlab/scripts/lian_validation_feasibility
r"""Can the Lian 2019 MAGIC furfural screen validate a chemogenomic model? A measurement.

THE PROPOSAL. Lian 2019 screened furfural tolerance genome-wide in the bAID host, the same
strain as the private Bioscreen runs, so it looks like an independent validation target for
a model trained to predict a compound's deletion profile. Three things decide whether it
can serve that role, and all three are measured here rather than argued.

1. REPRESENTABILITY. MAGIC carries three orthogonal modes: CRISPRa (activation), CRISPRi
   (interference) and CRISPRd (deletion). The models of experiments 031 to 040 carry a
   DELETION operator and nothing else, so only the CRISPRd third is representable at all.
   The script counts the guides and genes of each mode.

2. RELIABILITY of the Lian gene-level score. A screen cannot validate anything it does not
   itself measure reproducibly. Each gene carries several independent guides, so the
   guides are split at random into two halves, the halves are averaged per gene, and the
   two half-scores are correlated (repeated ``N_SPLITS`` times with different splits). The
   Spearman-Brown step gives the reliability of the full gene score. This is the ceiling
   on any correlation anything can have with it.

3. AGREEMENT with the deletion chemogenomics the models are trained on. The Vanacloig 2022
   furfural profile (``results/profile_matrix.csv``, the measured column) against the Lian
   gene score, per mode and per round, over the shared genes. Both carry the same polarity
   (positive = the perturbation helps under the compound: Vanacloig
   ``log2(inhibitor/control)``, positive = resistant deletion; Lian
   ``log2(furfural-selected/untreated)``, positive = tolerance), so agreement is a POSITIVE
   correlation. The attenuation ceiling ``sqrt(r_lian * r_vanacloig)`` is reported beside
   every correlation, with ``r_vanacloig`` the compound's ``measured_reliability`` from
   ``results/profile_predicted_vs_measured.csv``: a correlation far below its ceiling is a
   measured disagreement, one near a tiny ceiling is a statement about noise and nothing
   else.

Also scored: hit overlap at the top and bottom of both lists against the hypergeometric
null, since a genome-wide rank correlation on a mostly-noise screen is dominated by the
noise; and the rank of the paper's own validated hits (SAP30d round 1, SLX5i round 1,
PDR1i round 3) inside their own mode, which checks the reprocessed table rather than the
biology.

THE ROUNDS ARE AN EPISTASIS LADDER. Each round transforms the library into the strain
built from the previous round's best hit and raises the dose: round 1 screens the bAID
parent at 5 mM, round 2 the R1 parent (SIZ1i integrated) at 10 mM, round 3 the R2 parent
(SIZ1i + NAT1a integrated) at 15 mM. Only round 1 measures what a perturbation does on its
own in an unengineered parent, so only round 1 is comparable with single-perturbation
chemogenomics. The paper settles this with constructed strains rather than by inference
(Fig. 4): the round-2 and round-3 hits, alone or in combination, "marginally improved
furfural tolerance in the reference strain" and show a phenotype only in the SIZ1i
background. A round-2 or round-3 score is an effect CONDITIONAL on the integrated edits,
and reading it as a second opinion on the same question is a category error. The tables
carry ``comparable_to_single_perturbation_chemogenomics`` so the distinction cannot be
lost, and the later rounds are reported for completeness only. They are also not a dose
series, since dose and parent move together.

Outputs: ``results/lian_library.csv``, ``results/lian_reliability.csv``,
``results/lian_vs_vanacloig.csv``, ``results/lian_hit_overlap.csv``,
``results/lian_validated_hits.csv``, ``results/lian_feasibility_summary.json`` and one
figure in ``$ASSET_IMAGES_DIR/040-inhibitor-synergy-wetlab/``.

Run from the repository root:

    python experiments/040-inhibitor-synergy-wetlab/scripts/lian_validation_feasibility.py
"""

from __future__ import annotations

import json
import os
import os.path as osp
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy.stats import hypergeom, spearmanr

from torchcell.timestamp import timestamp
from torchcell.utils import (
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    mm_to_in,
    savefig_true_size_svg,
)

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
import mixture_rules as mr  # noqa: E402

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT = osp.dirname(osp.dirname(osp.abspath(__file__)))
RESULTS = osp.join(EXPERIMENT, "results")
IMAGES = osp.join(os.environ["ASSET_IMAGES_DIR"], "040-inhibitor-synergy-wetlab")
#: The reprocessed per-guide enrichment of the raw mirror (experiments/016; the per-guide
#: furfural enrichment was never released in any Nature supplement).
LIAN = osp.join(
    DATA_ROOT,
    "torchcell-raw",
    "lianMultifunctionalGenomewideCRISPR2019",
    "data",
    "guide_enrichment_final.tsv",
)
MODES = {
    "d": "CRISPRd (deletion)",
    "i": "CRISPRi (interference)",
    "a": "CRISPRa (activation)",
}
#: Representable by the models of 031 to 040, which carry a deletion operator only.
REPRESENTABLE = {"d"}
#: THE ROUNDS ARE AN EPISTASIS LADDER, NOT A DOSE SERIES AND NOT REPLICATES. Each round
#: transforms the library into the strain built from the previous round's best hit and
#: raises the dose, so round 1 is the only one that measures a perturbation on its own in
#: an unengineered parent. The paper demonstrates the difference with constructed strains
#: (Fig. 4): the round-2 and round-3 hits, alone or together, "marginally improved furfural
#: tolerance in the reference strain" and work only in the SIZ1i background. A round-2 or
#: round-3 score is therefore an effect CONDITIONAL on the integrated edits, and comparing
#: it with single-perturbation chemogenomics is a category error, not evidence.
ROUNDS = {
    "r1": "round 1, bAID parent, 5 mM furfural",
    "r2": "round 2, R1 parent (SIZ1i integrated), 10 mM furfural",
    "r3": "round 3, R2 parent (SIZ1i + NAT1a integrated), 15 mM furfural",
}
#: Only round 1 measures a perturbation in an unengineered parent.
COMPARABLE_ROUNDS = {"r1"}
#: The edits each round's parent already carries, and their modality. NAT1a is an
#: ACTIVATION, so the round-3 parent carries a gain-of-function the models cannot encode
#: either, on top of the interference.
PARENT_EDITS = {
    "r1": "",
    "r2": "SIZ1i (interference)",
    "r3": "SIZ1i (interference) + NAT1a (activation)",
}
#: The paper's own validated hits, as (gene, mode, round); a check on the reprocessing.
VALIDATED = [("SAP30", "d", "r1"), ("SLX5", "i", "r1"), ("PDR1", "i", "r3")]
COMPOUND = "furfural"
MEASURED_COLUMN = f"best_available|centered|{COMPOUND}"
RAW_COLUMN = f"best_available|raw|{COMPOUND}"
N_SPLITS = 20
TOP_K = (100, 200)
SEED = 0
ORANGE, RED, BLUE, GRAY = (
    PLOT_PALETTE[0],
    PLOT_PALETTE[1],
    PLOT_PALETTE[4],
    PLOT_PALETTE[5],
)

matplotlib.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 6,
        "axes.labelsize": 6,
        "axes.titlesize": 6,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "axes.linewidth": 0.5,
        "svg.fonttype": "none",
    }
)


def guides() -> pd.DataFrame:
    """The reprocessed per-guide table, controls and corrupted targets dropped."""
    d = pd.read_csv(LIAN, sep="\t")
    return d[(~d["is_control"]) & (~d["corrupted_gene"])].copy()


def library(d: pd.DataFrame) -> pd.DataFrame:
    """Guides, genes and guides per gene for each mode, with representability."""
    rows = []
    for mode, label in MODES.items():
        sub = d[d["mod"] == mode]
        per_gene = sub.groupby("gene").size()
        rows.append(
            {
                "mode": mode,
                "mode_label": label,
                "representable_by_a_deletion_operator": mode in REPRESENTABLE,
                "n_guides": int(len(sub)),
                "n_genes": int(sub["gene"].nunique()),
                "guides_per_gene_median": float(per_gene.median()),
                "guides_per_gene_min": int(per_gene.min()),
                "guides_per_gene_max": int(per_gene.max()),
            }
        )
    return pd.DataFrame(rows)


def gene_scores(d: pd.DataFrame, mode: str, rnd: str) -> pd.Series:
    """Mean guide enrichment per gene for one mode and round."""
    sub = d[d["mod"] == mode][["gene", f"{rnd}_log2fc_mean"]].dropna()
    return sub.groupby("gene")[f"{rnd}_log2fc_mean"].mean()


def reliability(d: pd.DataFrame) -> pd.DataFrame:
    """Split-half reliability of the gene score, per mode and round.

    The guides of a gene are split at random into two halves and each half is averaged;
    the correlation of the halves measures a HALF-length score, so Spearman-Brown
    (``2r / (1 + r)``) steps it up to the full gene score. Repeated over ``N_SPLITS``
    random splits; the spread across splits is reported.
    """
    rng = np.random.default_rng(SEED)
    rows = []
    for mode in MODES:
        for rnd in ROUNDS:
            sub = d[d["mod"] == mode][["gene", f"{rnd}_log2fc_mean"]].dropna()
            sub = sub.rename(columns={f"{rnd}_log2fc_mean": "value"})
            halves = []
            n_genes = 0
            for _ in range(N_SPLITS):
                sub = sub.assign(half=rng.integers(0, 2, len(sub)))
                wide = sub.groupby(["gene", "half"])["value"].mean().unstack("half")
                wide = wide.dropna()
                n_genes = len(wide)
                halves.append(float(spearmanr(wide[0], wide[1]).statistic))
            r = float(np.mean(halves))
            rows.append(
                {
                    "mode": mode,
                    "mode_label": MODES[mode],
                    "round": rnd,
                    "round_label": ROUNDS[rnd],
                    "parent_edits": PARENT_EDITS[rnd],
                    "measures_a_perturbation_alone": rnd in COMPARABLE_ROUNDS,
                    "n_genes_both_halves": n_genes,
                    "split_half_spearman": r,
                    "split_half_lo": float(np.percentile(halves, 2.5)),
                    "split_half_hi": float(np.percentile(halves, 97.5)),
                    "spearman_brown_reliability": float(2 * r / (1 + r)),
                }
            )
    return pd.DataFrame(rows)


def vanacloig_profile() -> tuple[pd.DataFrame, float]:
    """The measured Vanacloig furfural profile and its reliability."""
    matrix = pd.read_csv(osp.join(RESULTS, "profile_matrix.csv"))
    pvm = pd.read_csv(osp.join(RESULTS, "profile_predicted_vs_measured.csv"))
    hit = pvm[(pvm["compound"] == COMPOUND) & (pvm["target"] == "centered")]
    assert len(hit), f"{COMPOUND} has no measured row in profile_predicted_vs_measured"
    return matrix, float(hit["measured_reliability"].iloc[0])


def agreement(
    d: pd.DataFrame, matrix: pd.DataFrame, van_reliability: float, rel: pd.DataFrame
) -> pd.DataFrame:
    """Lian gene score against the measured Vanacloig furfural profile."""
    key = rel.set_index(["mode", "round"])["spearman_brown_reliability"]
    rows = []
    for mode in MODES:
        for rnd in ROUNDS:
            lian = gene_scores(d, mode, rnd)
            for column, target in ((RAW_COLUMN, "raw"), (MEASURED_COLUMN, "centered")):
                joined = (
                    matrix[["gene_name", column]]
                    .dropna()
                    .merge(lian.rename("lian"), left_on="gene_name", right_index=True)
                )
                stat = spearmanr(joined[column], joined["lian"])
                lian_rel = float(key.loc[(mode, rnd)])
                ceiling = (
                    float(np.sqrt(lian_rel * van_reliability))
                    if lian_rel > 0 and van_reliability > 0
                    else np.nan
                )
                rows.append(
                    {
                        "mode": mode,
                        "mode_label": MODES[mode],
                        "round": rnd,
                        "round_label": ROUNDS[rnd],
                        "parent_edits": PARENT_EDITS[rnd],
                        "comparable_to_single_perturbation_chemogenomics": (
                            rnd in COMPARABLE_ROUNDS
                        ),
                        "vanacloig_target": target,
                        "n_shared_genes": int(len(joined)),
                        "spearman": float(stat.statistic),
                        "p_value": float(stat.pvalue),
                        "lian_reliability": lian_rel,
                        "vanacloig_reliability": van_reliability,
                        "attenuation_ceiling": ceiling,
                        "fraction_of_ceiling": (
                            float(stat.statistic) / ceiling if ceiling > 0 else np.nan
                        ),
                    }
                )
    return pd.DataFrame(rows)


def hit_overlap(d: pd.DataFrame, matrix: pd.DataFrame) -> pd.DataFrame:
    """Overlap of the extreme tails of both lists against the hypergeometric null."""
    rows = []
    for mode in MODES:
        for rnd in ROUNDS:
            lian = gene_scores(d, mode, rnd)
            joined = (
                matrix[["gene_name", MEASURED_COLUMN]]
                .dropna()
                .merge(lian.rename("lian"), left_on="gene_name", right_index=True)
            )
            n = len(joined)
            for k in TOP_K:
                for lian_end in ("top", "bottom"):
                    for van_end in ("top", "bottom"):
                        a = set(
                            joined.nlargest(k, "lian")["gene_name"]
                            if lian_end == "top"
                            else joined.nsmallest(k, "lian")["gene_name"]
                        )
                        b = set(
                            joined.nlargest(k, MEASURED_COLUMN)["gene_name"]
                            if van_end == "top"
                            else joined.nsmallest(k, MEASURED_COLUMN)["gene_name"]
                        )
                        observed = len(a & b)
                        rows.append(
                            {
                                "mode": mode,
                                "round": rnd,
                                "k": k,
                                "lian_tail": lian_end,
                                "vanacloig_tail": van_end,
                                "n_shared_genes": n,
                                "overlap": observed,
                                "expected": k * k / n,
                                "p_value": float(hypergeom.sf(observed - 1, n, k, k)),
                            }
                        )
    return pd.DataFrame(rows)


def validated_hits(d: pd.DataFrame, matrix: pd.DataFrame) -> pd.DataFrame:
    """Where the paper's validated hits sit in the reprocessed table and in Vanacloig."""
    rows = []
    for gene, mode, rnd in VALIDATED:
        lian = gene_scores(d, mode, rnd)
        percentile = float(lian.rank(pct=True).get(gene, np.nan)) * 100
        van = matrix.loc[matrix["gene_name"] == gene, MEASURED_COLUMN]
        rows.append(
            {
                "gene": gene,
                "mode": mode,
                "mode_label": MODES[mode],
                "round": rnd,
                "lian_score": float(lian.get(gene, np.nan)),
                "lian_percentile_in_its_mode": percentile,
                "n_genes_in_mode": int(len(lian)),
                "in_vanacloig_profile": bool(len(van)),
                "vanacloig_centered": float(van.iloc[0]) if len(van) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def figure(
    lib: pd.DataFrame, rel: pd.DataFrame, agr: pd.DataFrame, van_reliability: float
) -> list[str]:
    """Three panels: library by mode, reliability by mode, agreement against its ceiling."""
    fig, axes = plt.subplots(
        1, 3, figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(52))
    )
    order = ["d", "i", "a"]
    colors = {"d": BLUE, "i": ORANGE, "a": GRAY}

    ax = axes[0]
    x = np.arange(len(order))
    ax.bar(
        x,
        [int(lib.loc[lib["mode"] == m, "n_genes"].iloc[0]) for m in order],
        color=[colors[m] for m in order],
        edgecolor="black",
        linewidth=0.5,
        width=0.65,
    )
    for xi, m in zip(x, order, strict=True):
        n = int(lib.loc[lib["mode"] == m, "n_guides"].iloc[0])
        ax.text(xi, 60, f"{n:,} guides", ha="center", va="bottom", rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels([MODES[m].split(" ")[0] for m in order])
    ax.set_ylabel("genes targeted")
    ax.set_title("every mode covers the genome; a quarter of guides delete")
    mr.box(ax)

    ax = axes[1]
    width = 0.26
    for i, rnd in enumerate(ROUNDS):
        sub = rel[rel["round"] == rnd].set_index("mode")
        ax.bar(
            x + (i - 1) * width,
            [sub.loc[m, "spearman_brown_reliability"] for m in order],
            width=width,
            color=[colors[m] for m in order],
            alpha=1.0 - 0.28 * i,
            edgecolor="black",
            linewidth=0.5,
            label=f"round {rnd[-1]}",
        )
    ax.axhline(
        van_reliability,
        color=RED,
        linewidth=1.0,
        linestyle="--",
        label="Vanacloig furfural profile",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([MODES[m].split(" ")[0] for m in order])
    ax.set_ylabel("reliability of the gene score")
    ax.set_ylim(-0.05, 1.0)
    ax.set_title("only the interference mode is reproducible")
    ax.legend(loc="upper left", frameon=False, ncol=2)
    mr.box(ax)
    mr.tenth_grid(ax, "y")

    ax = axes[2]
    sub = agr[(agr["vanacloig_target"] == "centered")]
    labels, values, ceilings, bar_colors, hatches = [], [], [], [], []
    for m in order:
        for rnd in ROUNDS:
            row = sub[(sub["mode"] == m) & (sub["round"] == rnd)].iloc[0]
            labels.append(f"{m}{rnd[-1]}")
            values.append(row["spearman"])
            ceilings.append(row["attenuation_ceiling"])
            bar_colors.append(colors[m])
            hatches.append("" if rnd in COMPARABLE_ROUNDS else "///")
    y = np.arange(len(labels))
    bars = ax.barh(
        y, values, color=bar_colors, edgecolor="black", linewidth=0.5, height=0.65
    )
    for bar, hatch in zip(bars, hatches, strict=True):
        bar.set_hatch(hatch)
    ax.plot(ceilings, y, marker="D", markersize=2.5, linestyle="none", color=RED)
    ax.axvline(0.0, color="black", linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Spearman against the Vanacloig furfural profile")
    ax.set_xlim(-0.1, 0.75)
    ax.set_title("round 1 reads zero against its ceiling")
    mr.box(ax)

    handles = [
        plt.Line2D([], [], marker="D", color=RED, linestyle="none", markersize=2.5),
        matplotlib.patches.Patch(
            facecolor="white", edgecolor="black", linewidth=0.5, hatch="///"
        ),
    ]
    fig.legend(
        handles,
        [
            "attenuation ceiling, sqrt(reliability of each side)",
            "epistatic round: the parent carries the earlier rounds' hits",
        ],
        loc="lower center",
        ncol=2,
        frameon=False,
    )
    fig.subplots_adjust(left=0.07, right=0.99, top=0.9, bottom=0.26, wspace=0.42)
    mr.panel_letters(fig, axes)
    os.makedirs(IMAGES, exist_ok=True)
    stem = osp.join(IMAGES, f"lian_validation_feasibility_{timestamp()}")
    savefig_true_size_svg(fig, stem + ".svg")
    fig.savefig(stem + ".png", dpi=300)
    plt.close(fig)
    return [stem + ".svg", stem + ".png"]


def main() -> None:
    d = guides()
    matrix, van_reliability = vanacloig_profile()

    lib = library(d)
    lib.to_csv(osp.join(RESULTS, "lian_library.csv"), index=False)
    print(lib.to_string(index=False))

    rel = reliability(d)
    rel.to_csv(osp.join(RESULTS, "lian_reliability.csv"), index=False)
    print()
    print(
        rel[
            [
                "mode",
                "round",
                "n_genes_both_halves",
                "split_half_spearman",
                "spearman_brown_reliability",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )

    agr = agreement(d, matrix, van_reliability, rel)
    agr.to_csv(osp.join(RESULTS, "lian_vs_vanacloig.csv"), index=False)
    print()
    print(
        agr[
            [
                "mode",
                "round",
                "vanacloig_target",
                "n_shared_genes",
                "spearman",
                "p_value",
                "attenuation_ceiling",
                "fraction_of_ceiling",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )

    overlap = hit_overlap(d, matrix)
    overlap.to_csv(osp.join(RESULTS, "lian_hit_overlap.csv"), index=False)
    print()
    print(
        overlap[overlap["mode"] == "d"][
            [
                "round",
                "k",
                "lian_tail",
                "vanacloig_tail",
                "overlap",
                "expected",
                "p_value",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )

    hits = validated_hits(d, matrix)
    hits.to_csv(osp.join(RESULTS, "lian_validated_hits.csv"), index=False)
    print()
    print(hits.round(3).to_string(index=False))

    deletion = agr[(agr["mode"] == "d") & (agr["vanacloig_target"] == "centered")]
    best = agr.loc[agr["spearman"].idxmax()]
    summary = {
        "generated_by": (
            "experiments/040-inhibitor-synergy-wetlab/scripts/lian_validation_feasibility.py"
        ),
        "generated_at": timestamp(),
        "lian_table": LIAN,
        "compound": COMPOUND,
        "vanacloig_measured_reliability": van_reliability,
        "representable_fraction_of_guides": float(
            lib.loc[lib["mode"].isin(REPRESENTABLE), "n_guides"].sum()
            / lib["n_guides"].sum()
        ),
        "deletion_mode_reliability_round1": float(
            rel[(rel["mode"] == "d") & (rel["round"] == "r1")][
                "spearman_brown_reliability"
            ].iloc[0]
        ),
        "interference_mode_reliability_round1": float(
            rel[(rel["mode"] == "i") & (rel["round"] == "r1")][
                "spearman_brown_reliability"
            ].iloc[0]
        ),
        "deletion_round1_spearman_vs_vanacloig": float(
            deletion[deletion["round"] == "r1"]["spearman"].iloc[0]
        ),
        "only_comparable_round": (
            "r1, the bAID parent; rounds 2 and 3 screen engineered parents and measure "
            "effects conditional on the integrated edits"
        ),
        "best_agreement_any_mode_round": {
            "mode": str(best["mode"]),
            "round": str(best["round"]),
            "target": str(best["vanacloig_target"]),
            "spearman": float(best["spearman"]),
            "attenuation_ceiling": float(best["attenuation_ceiling"]),
        },
        "n_shared_genes": int(deletion["n_shared_genes"].iloc[0]),
    }
    with open(osp.join(RESULTS, "lian_feasibility_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print()
    print(json.dumps(summary, indent=2))
    print("\n".join(figure(lib, rel, agr, van_reliability)))


if __name__ == "__main__":
    main()
