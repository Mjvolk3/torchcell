# experiments/010-kuzmin-tmi/scripts/metabolic_positive_predictions.py
# [[experiments.010-kuzmin-tmi.scripts.metabolic_positive_predictions]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/010-kuzmin-tmi/scripts/metabolic_positive_predictions
#
# Does inference_1 predict any positive interaction on a metabolic gene?
#
# The 601-gene inference_1 roster was narrowed from four source lists, one of which is
# the expanded metabolic set (1,316 genes of the 3,170 considered). The panel work so
# far has ranked the whole space without asking whether the metabolic subset appears in
# the positive tail at all, and that is a different question from "is the tail real":
# a set can be well represented in the roster and still absent from the top.
#
# This counts, over the full inference_1 space rather than a top-k slice:
#   - how many of the 601 roster genes and the realized inference space are metabolic
#   - the predicted-tau distribution of triples by metabolic content (0, 1, 2, 3 genes)
#   - how many triples clear each positive cut, split by metabolic content, and the
#     enrichment of each stratum against the base rate
#   - the same after the 50-distinct-screen gate, since single-screen genes carry most
#     of the raw positive tail
#
# Predicted tau is the mean over the three checkpoints, the same ensemble used
# throughout this experiment. A predicted value is not a measurement, so nothing here
# says an interaction exists; it says what the model asserts and for which genes.
#
# Run from repo root:
#   ~/miniconda3/envs/torchcell/bin/python \
#     experiments/010-kuzmin-tmi/scripts/metabolic_positive_predictions.py
#
# Outputs:
#   results/metabolic_positive_predictions.csv       per-stratum counts at each cut
#   results/metabolic_positive_gene_ranks.csv        per-gene positive rate, metabolic flagged
#   results/metabolic_positive_predictions.json      the counts quoted in prose
#   $ASSET_IMAGES_DIR/010-kuzmin-tmi/metabolic_positive_predictions.{png,svg}

import json
import os
import os.path as osp
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from matplotlib.ticker import MultipleLocator

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))

from torchcell.utils import (
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    mm_to_in,
    savefig_true_size_svg,
)

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
ASSET_IMAGES_DIR = os.environ["ASSET_IMAGES_DIR"]

RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "010-kuzmin-tmi", "results")
IMAGES_DIR = osp.join(ASSET_IMAGES_DIR, "010-kuzmin-tmi")
SELECTION_CSV = osp.join(
    EXPERIMENT_ROOT,
    "006-kuzmin-tmi",
    "results",
    "inference_preprocessing_expansion",
    "expanded_genes_analysis.csv",
)

# The cuts used elsewhere in this experiment: the Kuzmin 2020 symmetric positive call,
# the stringent tier, and Baryshnikova 2010's positive stringent arm.
CUTS = [0.08, 0.12, 0.16, 0.20]
MIN_DISTINCT_SCREENS = 50


def set_plot_style():
    """Applied inside the plotting function: dataset construction resets rcParams."""
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 6,
            "axes.labelsize": 6,
            "axes.titlesize": 6,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 5,
            "legend.title_fontsize": 5,
            "figure.titlesize": 6,
            "svg.fonttype": "none",
            "axes.linewidth": 0.5,
            "savefig.bbox": None,
            "savefig.pad_inches": 0.0,
        }
    )


def metabolic_genes() -> tuple[set[str], pd.DataFrame]:
    sel = pd.read_csv(SELECTION_CSV)
    return set(sel.loc[sel["in_expanded_metabolic"].astype(bool), "gene"]), sel


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)

    from positive_panel_selection import distinct_screens, load_space  # noqa: E402
    from positive_panel_selection import BUILD_DIR  # noqa: E402

    metabolic, sel = metabolic_genes()
    selected = set(sel.loc[sel["is_selected"].astype(bool), "gene"])
    print(f"selection table: {len(sel):,} genes considered, {len(selected)} selected, "
          f"{len(metabolic):,} flagged metabolic")
    print(f"metabolic AND selected: {len(metabolic & selected)}")

    triples, preds, vocab = load_space()
    tau = preds.mean(axis=1)
    vocab_arr = np.array(vocab)
    is_metabolic = np.isin(vocab_arr, list(metabolic))
    realized = set(vocab)
    print(f"inference_1 space: {len(tau):,} triples over {len(vocab)} realized genes, "
          f"{int(is_metabolic.sum())} of them metabolic")

    # Metabolic content of each triple: 0, 1, 2 or 3 of its genes.
    n_metabolic = is_metabolic[triples].sum(axis=1).astype(np.int8)
    print("\ntriples by metabolic gene count:")
    for k in range(4):
        m = n_metabolic == k
        print(f"  {k} metabolic genes: {int(m.sum()):>12,}  "
              f"({m.mean():6.2%})  max predicted tau {tau[m].max():+.4f}"
              if m.sum() else f"  {k} metabolic genes: 0")

    with open(osp.join(BUILD_DIR, "processed", "is_any_perturbed_gene_index.json")) as f:
        gene_index = json.load(f)
    screens = distinct_screens(gene_index)
    passes_gate = np.array([screens.get(g, 0) >= MIN_DISTINCT_SCREENS for g in vocab])
    gated = passes_gate[triples].all(axis=1)
    print(f"\ntriples surviving the {MIN_DISTINCT_SCREENS}-screen gate: "
          f"{int(gated.sum()):,} of {len(tau):,}")

    rows = []
    for gate_name, gate in (("all triples", np.ones(len(tau), bool)), ("screen-gated", gated)):
        base_total = int(gate.sum())
        for cut in CUTS:
            hit = gate & (tau > cut)
            n_hit = int(hit.sum())
            for k in range(4):
                stratum = gate & (n_metabolic == k)
                n_stratum = int(stratum.sum())
                n_both = int((hit & (n_metabolic == k)).sum())
                share_of_hits = n_both / n_hit if n_hit else np.nan
                share_of_space = n_stratum / base_total if base_total else np.nan
                rows.append(
                    {
                        "gate": gate_name,
                        "tau_cut": cut,
                        "n_metabolic_genes": k,
                        "n_triples_in_stratum": n_stratum,
                        "n_above_cut": n_both,
                        "rate_above_cut": n_both / n_stratum if n_stratum else np.nan,
                        "share_of_all_hits": share_of_hits,
                        "share_of_space": share_of_space,
                        "enrichment": (share_of_hits / share_of_space)
                        if share_of_space else np.nan,
                    }
                )
            rows.append(
                {
                    "gate": gate_name,
                    "tau_cut": cut,
                    "n_metabolic_genes": -1,  # any metabolic gene present
                    "n_triples_in_stratum": int((gate & (n_metabolic > 0)).sum()),
                    "n_above_cut": int((hit & (n_metabolic > 0)).sum()),
                    "rate_above_cut": (
                        int((hit & (n_metabolic > 0)).sum())
                        / max(int((gate & (n_metabolic > 0)).sum()), 1)
                    ),
                    "share_of_all_hits": (
                        int((hit & (n_metabolic > 0)).sum()) / n_hit if n_hit else np.nan
                    ),
                    "share_of_space": (
                        int((gate & (n_metabolic > 0)).sum()) / base_total
                        if base_total else np.nan
                    ),
                    "enrichment": np.nan,
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv(osp.join(RESULTS_DIR, "metabolic_positive_predictions.csv"), index=False)
    print("\n=== positive predictions by metabolic content ===")
    show = table[table["n_metabolic_genes"].isin([-1, 0])]
    print(show.to_string(index=False))

    # Per-gene positive rate, so a "which metabolic genes" question is answerable.
    per_gene = []
    for code, gene in enumerate(vocab):
        member = (triples == code).any(axis=1)
        n = int(member.sum())
        per_gene.append(
            {
                "gene": gene,
                "is_metabolic": bool(is_metabolic[code]),
                "distinct_screens": int(screens.get(gene, 0)),
                "n_triples": n,
                "mean_predicted_tau": float(tau[member].mean()) if n else np.nan,
                "max_predicted_tau": float(tau[member].max()) if n else np.nan,
                "frac_above_0.08": float((tau[member] > 0.08).mean()) if n else np.nan,
                "frac_above_0.16": float((tau[member] > 0.16).mean()) if n else np.nan,
            }
        )
    gene_df = pd.DataFrame(per_gene).sort_values("max_predicted_tau", ascending=False)
    gene_df.to_csv(
        osp.join(RESULTS_DIR, "metabolic_positive_gene_ranks.csv"), index=False
    )
    print("\ntop 15 genes by best predicted tau (metabolic flagged):")
    print(gene_df.head(15).to_string(index=False))

    met = gene_df[gene_df["is_metabolic"]]
    print(f"\nmetabolic genes in the realized space: {len(met)}")
    print("top 10 metabolic genes by best predicted tau:")
    print(met.head(10).to_string(index=False))

    summary = {
        "n_genes_considered": int(len(sel)),
        "n_genes_selected": len(selected),
        "n_metabolic_flagged": len(metabolic),
        "n_metabolic_selected": len(metabolic & selected),
        "n_metabolic_realized": int(is_metabolic.sum()),
        "n_realized_genes": len(vocab),
        "n_triples": int(len(tau)),
        "n_triples_with_any_metabolic": int((n_metabolic > 0).sum()),
        "n_triples_all_metabolic": int((n_metabolic == 3).sum()),
        "max_predicted_tau_overall": float(tau.max()),
        "max_predicted_tau_any_metabolic": float(tau[n_metabolic > 0].max()),
        "max_predicted_tau_all_metabolic": (
            float(tau[n_metabolic == 3].max()) if (n_metabolic == 3).any() else None
        ),
        "n_triples_screen_gated": int(gated.sum()),
        "max_predicted_tau_gated_any_metabolic": (
            float(tau[gated & (n_metabolic > 0)].max())
            if (gated & (n_metabolic > 0)).any() else None
        ),
        "counts_above_cut": {
            f"{cut}": {
                "all": int((tau > cut).sum()),
                "any_metabolic": int(((tau > cut) & (n_metabolic > 0)).sum()),
                "all_three_metabolic": int(((tau > cut) & (n_metabolic == 3)).sum()),
                "gated_any_metabolic": int(
                    ((tau > cut) & gated & (n_metabolic > 0)).sum()
                ),
            }
            for cut in CUTS
        },
    }
    with open(osp.join(RESULTS_DIR, "metabolic_positive_predictions.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\n" + json.dumps(summary, indent=2))

    plot(tau, n_metabolic, gated, gene_df,
         osp.join(IMAGES_DIR, "metabolic_positive_predictions"))
    print(f"\nwrote figures to {IMAGES_DIR}")


def plot(tau, n_metabolic, gated, gene_df, out_stem):
    set_plot_style()
    fig, axes = plt.subplots(
        1, 3,
        figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(56.0)),
        gridspec_kw={"width_ratios": [1.1, 1.0, 1.0]},
    )

    # A histogram buries the answer: every stratum piles up at zero and the tail that
    # decides the question is a few dozen triples. The survival curve shows exactly
    # where each stratum runs out, and the all-metabolic curve ends before the call cut.
    ax = axes[0]
    grid = np.linspace(0.0, float(tau.max()), 400)
    for k, color, label in (
        (0, PLOT_PALETTE[5], "no metabolic gene"),
        (1, PLOT_PALETTE[0], "1 metabolic gene"),
        (2, PLOT_PALETTE[1], "2 metabolic genes"),
        (3, PLOT_PALETTE[2], "3 metabolic genes"),
    ):
        m = n_metabolic == k
        if m.sum() == 0:
            continue
        sub = tau[m]
        surv = np.array([(sub > t).mean() for t in grid])
        ax.plot(grid, np.where(surv > 0, surv, np.nan), color=color, linewidth=0.9,
                label=f"{label} (n={int(m.sum()):,})", zorder=3)
        ax.plot([sub.max()], [1.0 / len(sub)], marker="o", markersize=2.6, color=color,
                markeredgecolor="black", markeredgewidth=0.25, linestyle="none", zorder=4)
    for c in CUTS:
        ax.axvline(c, color="black", linewidth=0.4, linestyle=":", zorder=2)
    ax.set_yscale("log")
    ax.set_xlabel("Predicted $\\tau$, three-checkpoint mean")
    ax.set_ylabel("Fraction of stratum above $\\tau$")
    ax.set_title("a  How far each stratum reaches\nfilled marker = the stratum's single best triple",
                 fontsize=6, loc="left", pad=3)
    ax.legend(loc="upper right", frameon=True, fontsize=5, handlelength=1.4,
              labelspacing=0.25, borderpad=0.3)

    ax = axes[1]
    width = 0.2
    xs = np.arange(len(CUTS))
    for j, (k, color, label) in enumerate((
        (0, PLOT_PALETTE[5], "0"),
        (1, PLOT_PALETTE[0], "1"),
        (2, PLOT_PALETTE[1], "2"),
        (3, PLOT_PALETTE[2], "3"),
    )):
        counts = [int((tau[n_metabolic == k] > c).sum()) for c in CUTS]
        rates = [float((tau[n_metabolic == k] > c).mean()) if (n_metabolic == k).any()
                 else 0.0 for c in CUTS]
        ax.bar(xs + (j - 1.5) * width, rates, width, color=color, edgecolor="black",
               linewidth=0.3, label=label, zorder=3)
        # A stratum with no hits draws no bar, and an absent bar reads as a plotting
        # fault rather than as the result. Label every count, zeros included.
        for xi, rate, n in zip(xs + (j - 1.5) * width, rates, counts):
            ax.annotate(f"{n}", (xi, rate if rate > 0 else 1e-7), fontsize=4.5,
                        ha="center", va="bottom", rotation=90, zorder=5,
                        color="black" if n else PLOT_PALETTE[1])
    ax.set_yscale("log")
    ax.set_ylim(1e-7, 1e-3)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"$\\tau>{c:+.2f}$" for c in CUTS])
    ax.set_ylabel("Fraction of stratum above the cut")
    ax.set_title("b  Positive rate by metabolic gene count", fontsize=6, loc="left", pad=3)
    ax.legend(title="metabolic genes", loc="upper right", frameon=True, fontsize=5,
              title_fontsize=5, handlelength=1.0, labelspacing=0.2, borderpad=0.3, ncols=4,
              columnspacing=0.8)

    ax = axes[2]
    met = gene_df[gene_df["is_metabolic"]]
    non = gene_df[~gene_df["is_metabolic"]]
    # Plotted as screens + 1: a log axis silently drops the zero-screen genes, and
    # those are exactly the ones the support gate exists to catch.
    ax.scatter(non["distinct_screens"] + 1, non["max_predicted_tau"], s=2.0,
               color="0.72", linewidths=0, zorder=3, label=f"other (n={len(non)})")
    ax.scatter(met["distinct_screens"] + 1, met["max_predicted_tau"], s=3.2,
               color=PLOT_PALETTE[0], linewidths=0, zorder=4,
               label=f"metabolic (n={len(met)})")
    ax.axvline(MIN_DISTINCT_SCREENS + 1, color="black", linewidth=0.5, linestyle=":",
               zorder=2)
    ax.set_xscale("log")
    ax.set_xlabel("Distinct query screens the gene was seen under, + 1")
    ax.set_ylabel("Best predicted $\\tau$ over its triples")
    ax.set_title("c  Best prediction against evidence", fontsize=6, loc="left", pad=3)
    ax.legend(loc="upper right", frameon=True, fontsize=5, handlelength=1.0,
              labelspacing=0.25, borderpad=0.3, scatterpoints=1)

    for ax in axes:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.5)
            spine.set_color("black")
        ax.grid(axis="y", which="major", color="0.85", linewidth=0.3, zorder=0)
        ax.set_axisbelow(True)
    axes[0].xaxis.set_major_locator(MultipleLocator(0.2))
    axes[0].xaxis.set_minor_locator(MultipleLocator(0.1))
    axes[0].tick_params(axis="x", which="minor", length=0)

    fig.suptitle(
        "Metabolic genes in the inference_1 positive tail. Predicted values, not "
        "measurements. The dotted line in c is the 50-screen support gate.",
        fontsize=6, y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(f"{out_stem}.png", dpi=300)
    savefig_true_size_svg(fig, f"{out_stem}.svg")
    plt.close(fig)


if __name__ == "__main__":
    main()
