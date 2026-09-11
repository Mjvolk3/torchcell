# experiments/026-metabolism-flux/scripts/plot_equilibrator_thermo.py
# [[experiments.026-metabolism-flux.scripts.plot_equilibrator_thermo]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/026-metabolism-flux/scripts/plot_equilibrator_thermo.py

r"""Panels for the eQuilibrator recomputation of yeast-GEM's thermodynamics.

Each panel answers one question about the replacement table:

``thermo_a_resolution``
    What fraction of metabolites resolved, and through which identifier namespace.
``thermo_b_distributions``
    How the formation energies are distributed, and where the reaction energies land.
``thermo_c_uncertainty``
    What the uncertainty actually is, which the shipped table does not carry at all.
``thermo_d_vs_shipped``
    Two panels. Left, the disagreement with the shipped table as published. Right, the
    same comparison with the shipped column converted from kcal and our table rebuilt at
    yETFL's compartment conditions, which is what separates a bookkeeping difference from
    a method difference.
``thermo_ef_diagnostics``
    Panels b and c of the compartment figure, whose panel a is the draw.io cell schematic
    composed alongside them in LaTeX. b is how much compartment-specific pH moves a
    reaction energy; c is whether summing our per-metabolite table reproduces
    eQuilibrator's own reaction-level call.

Panel c of that figure is a correctness check rather than a result: it tests OUR assembly,
the stoichiometric sum and the zero-proton convention, since the transform and the
formation energies are eQuilibrator's on both axes.

The unit conversion in ``thermo_d_vs_shipped`` and the matched-condition table it reads are
established in ``yeast9_delta_g_provenance.py``; run that script's ``--recompute`` mode
under the eQuilibrator environment before this one.
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
from matplotlib.colors import to_rgba

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
THERMO = osp.join(DATA_ROOT, "data", "torchcell", "thermo_equilibrator")
GEM_DB = osp.join(
    DATA_ROOT, "data/torchcell/yeast-GEM/yeast-GEM-9.0.2/data/databases"
)
RESULTS = osp.join(EXPERIMENT_ROOT, "026-metabolism-flux", "results")
OUT_DIR = osp.join(ASSET_IMAGES_DIR, "026-metabolism-flux")

# The GEM's two missing-value conventions. The sentinel is obvious; the literal NaN is
# not, and filtering only the sentinel leaves NaNs that propagate through every sum.
SENTINEL = 1e7

# The shipped columns are yETFL's, and yETFL's model declares thermo_units = 'kcal/mol'
# while the yeast9 paper reports them as kJ/mol. Measured in
# results/thermo_yeast9_provenance.json: regressing our kJ/mol formation energies on
# yETFL's over 1,927 shared metabolites gives a slope of 4.187 against 4.184.
KJ_PER_KCAL = 4.184


def apply_style() -> None:
    """Repo figure standards, from the shared source rather than a local copy.

    ``apply_paper_style`` carries the Arial mathtext settings, so a label mixing a
    symbol with words renders in one face at one size.
    """
    apply_paper_style()
    plt.rcParams.update({"xtick.major.width": 0.5, "ytick.major.width": 0.5})


def box(ax: plt.Axes) -> None:
    """All four spines visible, the repo's boxed look."""
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)


def read_shipped(path: str) -> dict[str, float]:
    """A GEM deltaG CSV with BOTH missing-value conventions rejected."""
    values: dict[str, float] = {}
    with open(path) as handle:
        next(handle)
        for line in handle:
            key, _, raw = line.strip().partition(",")
            try:
                value = float(raw)
            except ValueError:
                continue
            if value == SENTINEL or np.isnan(value):
                continue
            values[key] = value
    return values


def main() -> None:
    """Write the six panels plus a machine-readable summary of what they show."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-timestamp",
        action="store_true",
        help="Write stable filenames. The notes-tex build matches a figure by basename, "
        "so a timestamped panel would never be picked up.",
    )
    args = parser.parse_args()

    apply_style()
    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = "" if args.no_timestamp else f"_{timestamp()}"

    met = pd.read_parquet(osp.join(THERMO, "metabolite_dgf_prime.parquet"))
    rxn = pd.read_parquet(osp.join(THERMO, "reaction_drg_prime.parquet"))
    rxn_uniform = pd.read_parquet(
        osp.join(THERMO, "reaction_drg_prime_single_condition.parquet")
    )
    shipped = read_shipped(osp.join(GEM_DB, "model_rxnDeltaG.csv"))
    # The GEM itself is the only place that says which reactions cross a membrane.
    import cobra

    cobra_reactions = cobra.io.read_sbml_model(
        osp.join(DATA_ROOT, "data/torchcell/yeast-GEM/yeast-GEM-9.0.2/model/yeast-GEM.xml")
    ).reactions

    summary: dict[str, object] = {}

    # --- a. resolution coverage by identifier namespace -----------------------
    fig, ax = plt.subplots(
        figsize=(mm_to_in(PANEL_WIDTHS_MM["half"]), mm_to_in(45)), dpi=300
    )
    tiers = met["resolution_tier"].fillna("unresolved").value_counts()
    order = [t for t in
             ["metanetx.chemical", "chebi", "kegg.compound", "bigg.metabolite", "inchi",
              "unresolved"] if t in tiers.index]
    counts = [int(tiers[t]) for t in order]
    labels = [t.replace(".chemical", "").replace(".compound", "").replace(".metabolite", "")
              for t in order]
    colors = [PLOT_PALETTE[i] for i in range(len(order) - 1)] + [PLOT_PALETTE[5]]
    ax.bar(range(len(order)), counts, color=colors, edgecolor="black", linewidth=0.5)
    # Counts are NOT written over the bars; they are in the caption and in
    # results/thermo_plot_summary.json. Repo rule: only data inside the axes.
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("metabolites")
    ax.set_ylim(0, max(counts) * 1.18)
    box(ax)
    fig.tight_layout()
    savefig_true_size_svg(fig, osp.join(OUT_DIR, f"thermo_a_resolution{stamp}.svg"))
    fig.savefig(osp.join(OUT_DIR, f"thermo_a_resolution{stamp}.png"), dpi=300)
    plt.close(fig)
    summary["resolution_tiers"] = dict(zip(labels, counts))

    # --- b. formation and reaction energy distributions -----------------------
    fig, axes = plt.subplots(
        1, 2, figsize=(mm_to_in(PANEL_WIDTHS_MM["half_plus"]), mm_to_in(42)), dpi=300
    )
    dgf = met["dgf_prime_kj_per_mol"].dropna()
    axes[0].hist(dgf, bins=60, color=PLOT_PALETTE[0], edgecolor="black", linewidth=0.3)
    axes[0].set_xlabel(r"$\Delta_f G'^\circ$ (kJ mol$^{-1}$)")
    axes[0].set_ylabel("metabolites")
    drg = rxn.loc[rxn["all_participants_known"], "drg_prime_kj_per_mol"].dropna()
    # Clipped for display only; the tails are real but a few biomass-scale reactions
    # would otherwise compress every ordinary reaction into one bin.
    axes[1].hist(np.clip(drg, -400, 400), bins=60, color=PLOT_PALETTE[1],
                 edgecolor="black", linewidth=0.3)
    axes[1].set_xlabel(r"$\Delta_r G'^\circ$ (kJ mol$^{-1}$, clipped)")
    axes[1].set_ylabel("reactions")
    for ax in axes:
        box(ax)
    fig.tight_layout()
    savefig_true_size_svg(fig, osp.join(OUT_DIR, f"thermo_b_distributions{stamp}.svg"))
    fig.savefig(osp.join(OUT_DIR, f"thermo_b_distributions{stamp}.png"), dpi=300)
    plt.close(fig)
    summary["dgf_median"] = float(dgf.median())
    summary["drg_median"] = float(drg.median())

    # --- c. the uncertainty the shipped table does not have -------------------
    fig, ax = plt.subplots(
        figsize=(mm_to_in(PANEL_WIDTHS_MM["half"]), mm_to_in(42)), dpi=300
    )
    estimable = rxn[rxn["estimable"]]
    sigma = estimable["sigma_kj_per_mol"].dropna()
    nonzero = sigma[sigma >= 1e-9]
    ax.hist(nonzero, bins=50, color=PLOT_PALETTE[0], edgecolor="black", linewidth=0.3)
    ax.axvline(float(nonzero.median()), color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel(r"$\sigma(\Delta_r G'^\circ)$ (kJ mol$^{-1}$)")
    ax.set_ylabel("reactions")
    box(ax)
    fig.tight_layout()
    savefig_true_size_svg(fig, osp.join(OUT_DIR, f"thermo_c_uncertainty{stamp}.svg"))
    fig.savefig(osp.join(OUT_DIR, f"thermo_c_uncertainty{stamp}.png"), dpi=300)
    plt.close(fig)
    summary["n_estimable"] = int(len(estimable))
    summary["n_sigma_nonzero"] = int(len(nonzero))
    summary["n_sigma_zero_transport"] = int((sigma < 1e-9).sum())
    summary["sigma_median_nonzero"] = float(nonzero.median())

    # --- d. agreement with the shipped table, as published and once corrected -
    # Two panels of the same scatter, because the disagreement has two causes and only one
    # of them is about thermodynamics. LEFT is the comparison as the document made it: the
    # shipped column read as kJ/mol, our table at a uniform pH 7.0. RIGHT applies the two
    # corrections the provenance work established: the shipped column is yETFL's, in
    # kcal/mol, and yETFL priced every compartment at its own pH and ionic strength.
    # Transport has to be separated in both, not averaged in: at a single uniform pH a
    # reaction that moves a species between compartments has the SAME formation energy on
    # both sides, so its energy is exactly zero on the left by construction.
    fig, axes = plt.subplots(
        1, 2, figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(70)), dpi=300
    )
    ours = rxn_uniform[rxn_uniform["all_participants_known"]].set_index("reaction_id")[
        "drg_prime_kj_per_mol"
    ]
    matched_path = osp.join(THERMO, "reaction_drg_prime_yeast9_conditions.parquet")
    matched_frame = pd.read_parquet(matched_path)
    matched = matched_frame[matched_frame["all_participants_known"]].set_index(
        "reaction_id"
    )["drg_prime_kj_per_mol"]

    transport = {
        r.id for r in cobra_reactions if len({m.compartment for m in r.metabolites}) > 1
    }

    def scatter_panel(ax, ours_series, scale, xlabel, lim):
        """One agreement panel; returns its statistics on the chemical subset."""
        both = sorted(set(shipped) & set(ours_series.index))
        chemical = [k for k in both if k not in transport]
        moved = [k for k in both if k in transport]
        # Transport is the series the panel is about, so it takes a saturated primary and
        # sits on top; the chemical cloud stays orange underneath. Alpha is applied to the
        # face only, via to_rgba, so no edge or hatch is faded with it.
        ax.scatter(
            [shipped[k] * scale for k in chemical],
            [ours_series[k] for k in chemical],
            s=3.0, color=to_rgba(PLOT_PALETTE[0], 0.55), linewidths=0, zorder=2,
            label=f"single compartment ({len(chemical)})",
        )
        ax.scatter(
            [shipped[k] * scale for k in moved],
            [ours_series[k] for k in moved],
            s=3.0, color=to_rgba(PLOT_PALETTE[2], 0.55), linewidths=0, zorder=3,
            label=f"transport ({len(moved)})",
        )
        ax.plot(lim, lim, color="black", linewidth=0.6, linestyle="--", zorder=1)
        ax.axhline(0, color=PLOT_PALETTE[5], linewidth=0.4, zorder=1)
        ax.set_xlim(*lim)
        ax.set_ylim(*lim)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"eQuilibrator $\Delta_r G'^\circ$ (kJ mol$^{-1}$)")
        ax.set_aspect("equal")
        box(ax)
        x_chem = np.array([shipped[k] * scale for k in chemical])
        y_chem = np.array([ours_series[k] for k in chemical])
        x_moved = np.array([shipped[k] * scale for k in moved])
        y_moved = np.array([ours_series[k] for k in moved])
        return {
            "n_single_compartment": len(chemical),
            "n_transport": len(moved),
            "single_median_abs_diff": float(np.median(np.abs(y_chem - x_chem))),
            "single_pearson": float(np.corrcoef(x_chem, y_chem)[0, 1]),
            "single_spearman": float(
                pd.Series(x_chem).corr(pd.Series(y_chem), method="spearman")
            ),
            "transport_median_abs_diff": float(np.median(np.abs(y_moved - x_moved))),
            "transport_pearson": float(np.corrcoef(x_moved, y_moved)[0, 1]),
        }

    summary["vs_shipped_as_published"] = scatter_panel(
        axes[0],
        ours,
        1.0,
        "shipped " + r"$\Delta_r G'^\circ$ as labeled (kJ mol$^{-1}$)",
        (-220, 220),
    )
    summary["vs_shipped_corrected"] = scatter_panel(
        axes[1],
        matched,
        KJ_PER_KCAL,
        "shipped " + r"$\Delta_r G'^\circ \times 4.184$ (kJ mol$^{-1}$)",
        (-400, 400),
    )
    axes[0].legend(loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=2,
                   frameon=False, fontsize=6, markerscale=2.6,
                   handletextpad=0.2, borderpad=0.1, labelspacing=0.3)
    for letter, ax in zip("ab", axes):
        ax.text(-0.17, 1.06, letter, transform=ax.transAxes, fontsize=8,
                fontweight="bold", va="top")
    fig.tight_layout(w_pad=2.4)
    savefig_true_size_svg(fig, osp.join(OUT_DIR, f"thermo_d_vs_shipped{stamp}.svg"))
    fig.savefig(osp.join(OUT_DIR, f"thermo_d_vs_shipped{stamp}.png"), dpi=300)
    plt.close(fig)

    # --- b + c of the compartment figure --------------------------------------
    # Panel a of that figure is the draw.io cell schematic, composed in LaTeX, so this
    # file writes b and c and letters them accordingly. b is what a compartment-specific
    # pH does to a reaction energy; c is the check that our per-metabolite table, summed
    # over stoichiometry, reproduces eQuilibrator's own reaction-level call.
    validation_path = osp.join(RESULTS, "equilibrator_api_validation.json")
    checks = None
    if osp.exists(validation_path):
        with open(validation_path) as handle:
            checks = json.load(handle)

    fig, axes = plt.subplots(
        1, 2, figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(58)), dpi=300
    )

    ax = axes[0]
    covered = rxn["all_participants_known"] & rxn_uniform["all_participants_known"]
    shift = (
        rxn.loc[covered, "drg_prime_kj_per_mol"].to_numpy()
        - rxn_uniform.loc[covered, "drg_prime_kj_per_mol"].to_numpy()
    )
    moved = np.abs(shift[np.abs(shift) >= 0.01])
    ax.hist(np.clip(moved, 0, 300), bins=50, color=PLOT_PALETTE[0],
            edgecolor="black", linewidth=0.3)
    ax.axvline(float(np.median(moved)), color="black", linewidth=0.9, linestyle="--")
    ax.set_xlabel(r"$|\Delta|$ from compartment pH (kJ mol$^{-1}$, clipped at 300)")
    ax.set_ylabel("reactions")
    box(ax)
    summary["compartment_ph_n_moved"] = int(len(moved))
    summary["compartment_ph_median_shift"] = float(np.median(moved))
    summary["compartment_ph_p90_shift"] = float(np.percentile(moved, 90))

    ax = axes[1]
    if checks is not None:
        table = pd.DataFrame(checks["reactions"])
        ax.scatter(table["api"], table["ours"], s=7, color=PLOT_PALETTE[1],
                   edgecolors="black", linewidths=0.3, zorder=3)
        span = (min(table["api"].min(), table["ours"].min()) - 20,
                max(table["api"].max(), table["ours"].max()) + 20)
        ax.plot(span, span, color="black", linewidth=0.6, linestyle="--", zorder=1)
        ax.set_xlim(*span)
        ax.set_ylim(*span)
        ax.set_aspect("equal")
        ax.set_xlabel(r"eQuilibrator $\Delta_r G'^\circ$, its own call (kJ mol$^{-1}$)")
        ax.set_ylabel(r"our $\sum_i S_{ij}\,\Delta_f G'^\circ_i$ (kJ mol$^{-1}$)")
        summary["validation_max_abs_diff"] = checks["max_abs_diff"]
    box(ax)

    for letter, ax in zip("bc", axes):
        ax.text(-0.17, 1.06, letter, transform=ax.transAxes, fontsize=8,
                fontweight="bold", va="top")

    fig.tight_layout(w_pad=2.4)
    savefig_true_size_svg(fig, osp.join(OUT_DIR, f"thermo_ef_diagnostics{stamp}.svg"))
    fig.savefig(osp.join(OUT_DIR, f"thermo_ef_diagnostics{stamp}.png"), dpi=300)
    plt.close(fig)

    with open(osp.join(RESULTS, "thermo_plot_summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"panels written to {OUT_DIR}{stamp and ' with stamp ' + stamp}")


if __name__ == "__main__":
    main()
