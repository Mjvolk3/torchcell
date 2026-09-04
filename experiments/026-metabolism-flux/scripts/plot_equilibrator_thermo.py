# experiments/026-metabolism-flux/scripts/plot_equilibrator_thermo.py
# [[experiments.026-metabolism-flux.scripts.plot_equilibrator_thermo]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/026-metabolism-flux/scripts/plot_equilibrator_thermo.py

r"""Panels for the eQuilibrator recomputation of yeast-GEM's thermodynamics.

Six panels, each answering one question about the replacement table:

a. What fraction of metabolites resolved, and through which identifier namespace.
b. How the formation energies are distributed, and where the reaction energies land.
c. What the uncertainty actually is, which the shipped table does not carry at all.
d. Whether the recomputation agrees with the shipped table on the reactions both cover.
e. How much compartment-specific pH moves a reaction energy.
f. Whether summing formation energies reproduces eQuilibrator's own reaction call.

Panel f is the correctness check rather than a result: if it did not sit on the diagonal
to machine precision, nothing else on this page would mean anything.
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
from matplotlib.ticker import MultipleLocator

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
    met_uniform = pd.read_parquet(
        osp.join(THERMO, "metabolite_dgf_prime_single_condition.parquet")
    )
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
    bars = ax.bar(range(len(order)), counts, color=colors, edgecolor="black", linewidth=0.5)
    for rect, value in zip(bars, counts):
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 25, str(value),
                ha="center", va="bottom", fontsize=5)
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
    ax.hist(nonzero, bins=50, color=PLOT_PALETTE[2], edgecolor="black", linewidth=0.3)
    ax.axvline(float(nonzero.median()), color="black", linewidth=0.8, linestyle="--")
    ax.text(float(nonzero.median()), ax.get_ylim()[1] * 0.92,
            f"  median {nonzero.median():.2f}", fontsize=5, va="top")
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

    # --- d. agreement with the shipped table, at uniform pH -------------------
    fig, ax = plt.subplots(
        figsize=(mm_to_in(PANEL_WIDTHS_MM["half"]), mm_to_in(58)), dpi=300
    )
    ours = rxn_uniform[rxn_uniform["all_participants_known"]].set_index("reaction_id")[
        "drg_prime_kj_per_mol"
    ]
    # Transport has to be separated, not averaged in. At a single uniform pH a reaction
    # that moves a species between compartments has the SAME formation energy on both
    # sides, so its energy is exactly zero here by construction, while the shipped table
    # carries a nonzero number for it. Pooling the two makes the comparison look worse
    # than the chemistry warrants and hides which reactions actually disagree.
    transport = {
        r.id for r in cobra_reactions if len({m.compartment for m in r.metabolites}) > 1
    }
    both = sorted(set(shipped) & set(ours.index))
    chemical = [k for k in both if k not in transport]
    moved_only = [k for k in both if k in transport]

    for keys, color, label in (
        (chemical, PLOT_PALETTE[4], "chemical"),
        (moved_only, PLOT_PALETTE[5], "transport"),
    ):
        ax.scatter(
            [shipped[k] for k in keys],
            [ours[k] for k in keys],
            s=1.5,
            color=color,
            alpha=0.5,
            linewidths=0,
            label=f"{label} ({len(keys)})",
        )
    lim = (-450, 450)
    ax.plot(lim, lim, color="black", linewidth=0.5, linestyle="--")
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_xlabel(r"shipped $\Delta_r G'^\circ$ (kJ mol$^{-1}$)")
    ax.set_ylabel(r"eQuilibrator $\Delta_r G'^\circ$ (kJ mol$^{-1}$)")
    ax.set_aspect("equal")
    chemical_difference = np.abs(
        np.array([ours[k] for k in chemical]) - np.array([shipped[k] for k in chemical])
    )
    all_difference = np.abs(
        np.array([ours[k] for k in both]) - np.array([shipped[k] for k in both])
    )
    ax.text(0.04, 0.96,
            f"chemical median |diff|\n{np.median(chemical_difference):.2f}"
            r" kJ mol$^{-1}$",
            transform=ax.transAxes, fontsize=5, va="top")
    ax.legend(loc="lower right", frameon=False, fontsize=5, markerscale=3,
              handletextpad=0.2, borderpad=0.1)
    box(ax)
    fig.tight_layout()
    savefig_true_size_svg(fig, osp.join(OUT_DIR, f"thermo_d_vs_shipped{stamp}.svg"))
    fig.savefig(osp.join(OUT_DIR, f"thermo_d_vs_shipped{stamp}.png"), dpi=300)
    plt.close(fig)
    summary["vs_shipped_n"] = len(both)
    summary["vs_shipped_n_chemical"] = len(chemical)
    summary["vs_shipped_n_transport"] = len(moved_only)
    summary["vs_shipped_median_abs_diff_all"] = float(np.median(all_difference))
    summary["vs_shipped_median_abs_diff_chemical"] = float(
        np.median(chemical_difference)
    )

    # --- e. what compartment pH does ------------------------------------------
    fig, ax = plt.subplots(
        figsize=(mm_to_in(PANEL_WIDTHS_MM["half"]), mm_to_in(42)), dpi=300
    )
    covered = rxn["all_participants_known"] & rxn_uniform["all_participants_known"]
    shift = (
        rxn.loc[covered, "drg_prime_kj_per_mol"].to_numpy()
        - rxn_uniform.loc[covered, "drg_prime_kj_per_mol"].to_numpy()
    )
    moved = np.abs(shift[np.abs(shift) >= 0.01])
    ax.hist(np.clip(moved, 0, 300), bins=50, color=PLOT_PALETTE[3],
            edgecolor="black", linewidth=0.3)
    ax.axvline(float(np.median(moved)), color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel(r"$|\Delta|$ from compartment pH (kJ mol$^{-1}$, clipped)")
    ax.set_ylabel("reactions")
    ax.text(0.35, 0.9,
            f"{len(moved)} of {int(covered.sum())} move\nmedian {np.median(moved):.1f}",
            transform=ax.transAxes, fontsize=5, va="top")
    box(ax)
    fig.tight_layout()
    savefig_true_size_svg(fig, osp.join(OUT_DIR, f"thermo_e_compartment_ph{stamp}.svg"))
    fig.savefig(osp.join(OUT_DIR, f"thermo_e_compartment_ph{stamp}.png"), dpi=300)
    plt.close(fig)
    summary["compartment_ph_n_moved"] = int(len(moved))
    summary["compartment_ph_median_shift"] = float(np.median(moved))

    # --- f. the correctness check ---------------------------------------------
    validation_path = osp.join(RESULTS, "equilibrator_api_validation.json")
    if osp.exists(validation_path):
        with open(validation_path) as handle:
            checks = json.load(handle)
        fig, ax = plt.subplots(
            figsize=(mm_to_in(PANEL_WIDTHS_MM["half"]), mm_to_in(58)), dpi=300
        )
        table = pd.DataFrame(checks["reactions"])
        ax.scatter(table["api"], table["ours"], s=6, color=PLOT_PALETTE[1],
                   edgecolors="black", linewidths=0.3)
        span = (min(table["api"].min(), table["ours"].min()) - 20,
                max(table["api"].max(), table["ours"].max()) + 20)
        ax.plot(span, span, color="black", linewidth=0.5, linestyle="--")
        ax.set_xlim(*span)
        ax.set_ylim(*span)
        ax.set_aspect("equal")
        ax.set_xlabel(r"eQuilibrator API $\Delta_r G'^\circ$ (kJ mol$^{-1}$)")
        ax.set_ylabel(r"our $\sum_i S_{ij}\,\Delta_f G'^\circ_i$ (kJ mol$^{-1}$)")
        ax.text(0.04, 0.95,
                f"n = {len(table)}\nmax |diff| {checks['max_abs_diff']:.1e}",
                transform=ax.transAxes, fontsize=5, va="top")
        box(ax)
        fig.tight_layout()
        savefig_true_size_svg(fig, osp.join(OUT_DIR, f"thermo_f_validation{stamp}.svg"))
        fig.savefig(osp.join(OUT_DIR, f"thermo_f_validation{stamp}.png"), dpi=300)
        plt.close(fig)
        summary["validation_max_abs_diff"] = checks["max_abs_diff"]

    with open(osp.join(RESULTS, "thermo_plot_summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"panels written to {OUT_DIR}{stamp and ' with stamp ' + stamp}")


if __name__ == "__main__":
    main()
