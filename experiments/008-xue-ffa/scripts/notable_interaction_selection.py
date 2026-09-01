# experiments/008-xue-ffa/scripts/notable_interaction_selection.py
# [[experiments.008-xue-ffa.scripts.notable_interaction_selection]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/008-xue-ffa/scripts/notable_interaction_selection
#
# WHICH interactions are worth plotting, and WHY. This is the single definition of
# "notable" for 008, and the overlay builds read their target list from the CSV it emits
# rather than rendering every combination and pruning afterwards.
#
# The problem it solves: rendering every (model x readout x graph x topology x
# enrichment) combination produces 671 overlays totalling 3.9 GB, of which 588 are the
# `unenriched` complement and are by construction the NOT-notable ones. Selecting up front
# is both cheaper and honest, because the selection rule is then a committed artifact
# instead of a post-hoc choice of which renders to keep.
#
# SELECTION RULE, in three layers. Each layer is a separate, inspectable column so a
# reader can see exactly why any interaction was kept or dropped.
#
#   Layer 1, CONSENSUS GATE (must pass all):
#     - FDR < 0.05 within the plotted readout, in EVERY one of the four epistasis models
#     - the sign of the interaction agrees across all four models
#     Rationale: a claim that survives the multiplicative, additive, GLM log-link and
#     log-OLS nulls is a statement about the data, not about a modeling choice. This is
#     the model-independent evidence, and it is what the consensus panel shows.
#
#   Layer 2, EXTREMES (the ranking within the gate):
#     - rank by |tau|, taken SEPARATELY within positives and negatives
#     Rationale: pooling the ranks erases positives, since only 27 of 120 trigenic
#     interactions are positive on the linear scale. Ranking within sign keeps both
#     directions visible even when one is rare.
#
#   Layer 3, BUILD AXES (which plots get rendered at all):
#     - enrichment: `enriched` only
#     - readout: Total Titer, plus any species readout carrying its own notable set
#     - graph: only graphs the notable set actually overlaps
#
# SCALE ROBUSTNESS is carried as a flag, not a filter. A positive interaction on FFA titer
# is a statement about the scale it was measured on: 15 of 27 linear-positive triples go
# non-positive on the log scale, and no positive interaction reaches FDR in either
# log-scale model. Positives therefore never pass the Layer 1 consensus gate, and any
# positive shown is labeled as linear-scale-only.

import os
import os.path as osp
import re

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from statsmodels.stats.multitest import multipletests

import torchcell
from torchcell.utils import (
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    mm_to_in,
    savefig_true_size_svg,
)

load_dotenv()
plt.style.use(osp.join(osp.dirname(torchcell.__file__), "torchcell.mplstyle"))
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
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "axes.linewidth": 0.5,
        "lines.linewidth": 0.7,
        "savefig.bbox": "standard",
    }
)

EXPERIMENT_ROOT = os.getenv("EXPERIMENT_ROOT")
ASSET_IMAGES_DIR = os.getenv("ASSET_IMAGES_DIR")
RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "008-xue-ffa/results")
IMAGES_DIR = osp.join(ASSET_IMAGES_DIR, "008-xue-ffa")

MODELS = {
    "multiplicative": "multiplicative_trigenic_interactions_3_delta_normalized.csv",
    "additive": "additive_trigenic_interactions_3_delta_normalized.csv",
    "glm_log_link": "glm_log_link/glm_log_link_trigenic_interactions.csv",
    "log_ols": "glm_models/log_ols_trigenic_interactions.csv",
}
# The five individually measured free fatty acid species, plus their sum. "Total Titer" is
# the sum formed per replicate before averaging; it is not a sixth measurement.
SPECIES = ["C14:0", "C16:0", "C18:0", "C16:1", "C18:1"]
TOTAL = "Total Titer"
N_TOP = 8

C_POS = PLOT_PALETTE[0]
C_NEG = PLOT_PALETTE[1]
C_ACCENT = PLOT_PALETTE[2]
C_GRAY = PLOT_PALETTE[5]


def canonical(gene_set):
    """Model families write different separators, so joins need a canonical key."""
    return "-".join(sorted(re.split(r"[_:|-]", str(gene_set))))


def canonical_readout(label):
    """Readout label to a canonical form.

    Same class of defect as the gene-set separators: the linear-scale scripts write
    `C14:0` and the regression scripts write `C140`, so a cross-model join on the species
    readouts silently matches nothing. Only `Total Titer` agrees between the two families,
    which is why this went unnoticed while the work was confined to that readout.
    """
    return str(label).replace(":", "").replace(" ", "_")


def load_model(rel, readout):
    """One model's trigenic table for one readout, with within-readout BH added.

    The stored fdr_corrected_p pools all six readouts (714 tests). A figure about one
    readout is a family of that readout's tests, so BH is recomputed here at the family
    the figure actually claims over.
    """
    df = pd.read_csv(osp.join(RESULTS_DIR, rel))
    df["readout_key"] = df["ffa_type"].map(canonical_readout)
    df = df[(df["readout_key"] == canonical_readout(readout)) & df["p_value"].notna()].copy()
    if df.empty:
        raise ValueError(f"{rel}: no testable rows for readout {readout!r}")
    df["triple"] = df["gene_set"].map(canonical)
    reject, fdr, _, _ = multipletests(df["p_value"], method="fdr_bh", alpha=0.05)
    df["fdr_within"] = fdr
    df["fdr_sig"] = reject
    return df[["triple", "interaction_score", "p_value", "fdr_within", "fdr_sig"]]


def consensus_table(readout):
    """Merge all four models for one readout and apply the Layer 1 gate."""
    merged = None
    for name, rel in MODELS.items():
        m = load_model(rel, readout).rename(
            columns={
                "interaction_score": f"tau_{name}",
                "p_value": f"p_{name}",
                "fdr_within": f"fdr_{name}",
                "fdr_sig": f"sig_{name}",
            }
        )
        merged = m if merged is None else merged.merge(m, on="triple", how="inner")

    tau_cols = [f"tau_{m}" for m in MODELS]
    sig_cols = [f"sig_{m}" for m in MODELS]
    merged["n_models_sig"] = merged[sig_cols].sum(axis=1)
    signs = np.sign(merged[tau_cols].to_numpy())
    merged["sign_concordant"] = (np.abs(signs.sum(axis=1)) == len(MODELS))
    merged["consensus"] = (merged["n_models_sig"] == len(MODELS)) & merged["sign_concordant"]
    merged["direction"] = np.where(merged["tau_multiplicative"] > 0, "positive", "negative")
    # Rank by the multiplicative tau, the estimand that matches Kuzmin's tau-SGA.
    merged["abs_tau"] = merged["tau_multiplicative"].abs()
    merged["readout"] = readout
    return merged


def select_notable(cons):
    """Layer 2: rank within sign, inside the consensus gate."""
    gated = cons[cons["consensus"]].copy()
    gated["rank_in_sign"] = (
        gated.groupby("direction")["abs_tau"].rank(ascending=False, method="first")
    )
    gated["notable"] = gated["rank_in_sign"] <= N_TOP
    return gated.sort_values(["direction", "rank_in_sign"])


def panel_consensus(ax, cons):
    """How many models call each interaction, and in which direction."""
    counts = (
        cons.groupby(["n_models_sig", "direction"]).size().unstack(fill_value=0)
    ).reindex(range(0, len(MODELS) + 1), fill_value=0)
    for col in ("negative", "positive"):
        if col not in counts:
            counts[col] = 0
    x = np.arange(len(counts))
    ax.bar(x, counts["negative"], color=C_NEG, edgecolor="black", linewidth=0.4,
           label="negative $\\tau$")
    ax.bar(x, counts["positive"], bottom=counts["negative"], color=C_POS,
           edgecolor="black", linewidth=0.4, label="positive $\\tau$")
    for xi, (neg, pos) in enumerate(zip(counts["negative"], counts["positive"])):
        if neg + pos:
            ax.text(xi, neg + pos + 1.5, f"{int(neg + pos)}", ha="center", fontsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in counts.index])
    ax.set_xlabel("number of models calling FDR < 0.05")
    ax.set_ylabel("trigenic interactions")
    n_cons = int(cons["consensus"].sum())
    n_pos_cons = int((cons["consensus"] & (cons["direction"] == "positive")).sum())
    ax.set_title(
        f"{n_cons}/{len(cons)} called by all 4 models\n"
        f"{n_pos_cons} of them positive",
        fontsize=6, pad=3,
    )
    ax.legend(loc="upper center", frameon=False, handlelength=1.2, fontsize=5)


def panel_readouts(ax, per_readout):
    """Consensus interactions per readout, split by sign.

    The story here is an inversion, and it is the opposite of what the pre-df-fix numbers
    suggested. Total titer is the readout with NO positive consensus interaction at all,
    while C14:0 carries 20 and every one of them is positive. Summing the five species
    into a total is what removes the positive signal.
    """
    order = [TOTAL] + SPECIES
    neg = [per_readout[r]["cons_neg"] for r in order]
    pos = [per_readout[r]["cons_pos"] for r in order]
    x = np.arange(len(order))
    ax.bar(x, neg, color=C_NEG, edgecolor="black", linewidth=0.4,
           label="negative $\\tau$")
    ax.bar(x, pos, bottom=neg, color=C_POS, edgecolor="black", linewidth=0.4,
           label="positive $\\tau$")
    for xi, (n, p) in enumerate(zip(neg, pos)):
        if n + p:
            ax.text(xi, n + p + 1.6, f"{n + p}", ha="center", fontsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels(["Total\ntiter"] + [s.replace(":", ":\n") for s in SPECIES])
    ax.set_ylabel("consensus interactions (all four models)")
    n_pos_any = sum(v["cons_pos"] for v in per_readout.values())
    ax.set_title(
        f"{n_pos_any} positive consensus interactions\n"
        "exist, none on the summed total",
        fontsize=6, pad=3,
    )
    # Parked over the C16:1 column, the only place with headroom.
    ax.legend(loc="center", bbox_to_anchor=(0.80, 0.42), frameon=False,
              handlelength=1.2, fontsize=5)


def panel_top(ax, notable):
    """The selected interactions, with every model's tau shown for each."""
    sel = notable[notable["notable"]].sort_values("tau_multiplicative")
    y = np.arange(len(sel))
    marks = [("multiplicative", "o"), ("additive", "s"), ("glm_log_link", "^"),
             ("log_ols", "D")]
    for (name, mk), color in zip(marks, [PLOT_PALETTE[i] for i in (0, 1, 2, 4)]):
        ax.scatter(sel[f"tau_{name}"], y, s=9, marker=mk, facecolor=color,
                   edgecolor="black", linewidth=0.25, label=name, zorder=3)
    for yi, (_, row) in enumerate(sel.iterrows()):
        vals = [row[f"tau_{m}"] for m in MODELS]
        ax.plot([min(vals), max(vals)], [yi, yi], color=C_GRAY, linewidth=0.5, zorder=2)
    ax.axvline(0, color="black", linewidth=0.5, linestyle="--", zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels([t.replace("-", "–") for t in sel["triple"]], fontsize=5)
    ax.set_xlabel("$\\tau$ (trigenic interaction)")
    ax.set_title(
        f"Top {len(sel)} by |$\\tau$|, all 4 models\n"
        "each negative and sign-concordant",
        fontsize=6, pad=3,
    )
    ax.legend(loc="lower right", frameon=False, handlelength=1.0, fontsize=5)


def main():
    os.makedirs(IMAGES_DIR, exist_ok=True)

    cons_total = consensus_table(TOTAL)
    notable = select_notable(cons_total)

    per_readout = {}
    for r in [TOTAL] + SPECIES:
        c = consensus_table(r)
        cons = c[c["consensus"]]
        per_readout[r] = {
            "cons_pos": int((cons["direction"] == "positive").sum()),
            "cons_neg": int((cons["direction"] == "negative").sum()),
            "n_total": len(c),
        }

    print(f"Total-titer trigenic interactions: {len(cons_total)}")
    print(f"  consensus (FDR<0.05 in all 4 + sign concordant): "
          f"{int(cons_total['consensus'].sum())}")
    print(f"  of those positive: "
          f"{int((cons_total['consensus'] & (cons_total.direction == 'positive')).sum())}")
    print(f"  selected as notable (top {N_TOP} per sign): {int(notable['notable'].sum())}")
    print("\nper-readout CONSENSUS across all four models (negative / positive):")
    for r, v in per_readout.items():
        print(f"  {r:<12} {v['cons_neg']:>3} / {v['cons_pos']:>3}")
    tot_pos = sum(v["cons_pos"] for v in per_readout.values())
    print(f"  positive consensus interactions on ANY readout: {tot_pos}")

    sel_path = osp.join(RESULTS_DIR, "notable_interaction_selection.csv")
    notable.to_csv(sel_path, index=False)
    print(f"\nwrote {sel_path}")

    # One file per panel, each at a standard panel width and with NO panel letter.
    # Arrangement and A/B/C lettering are draw.io decisions, so baking either one in here
    # would remove the flexibility that storyboarding needs. Three "third" panels tile
    # across a 179 mm page (3 x 57.8 = 173.4 mm) if they end up in one row.
    panels = [
        ("consensus_agreement", "third", 62.0, lambda ax: panel_consensus(ax, cons_total)),
        ("readout_sign", "third", 62.0, lambda ax: panel_readouts(ax, per_readout)),
        ("top_interactions", "third", 62.0, lambda ax: panel_top(ax, notable)),
    ]
    for name, width_key, height_mm, draw in panels:
        fig, ax = plt.subplots(
            figsize=(mm_to_in(PANEL_WIDTHS_MM[width_key]), mm_to_in(height_mm))
        )
        draw(ax)
        ax.grid(axis="both", which="major", color="0.88", linewidth=0.3, zorder=0)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.5)
            spine.set_color("black")
        fig.tight_layout(pad=0.4)
        stem = osp.join(IMAGES_DIR, f"panel_{name}")
        fig.savefig(f"{stem}.png", dpi=300)
        savefig_true_size_svg(fig, f"{stem}.svg")
        plt.close(fig)
        print(f"wrote {stem}.svg  ({PANEL_WIDTHS_MM[width_key]} x {height_mm} mm)")


if __name__ == "__main__":
    main()
