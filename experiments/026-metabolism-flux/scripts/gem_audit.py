# experiments/026-metabolism-flux/scripts/gem_audit.py
# [[experiments.026-metabolism-flux.scripts.gem_audit]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/026-metabolism-flux/scripts/gem_audit.py

r"""Audit what the enzyme-constrained thermodynamic layer actually rests on.

Run from the repo/worktree root::

    PYTHONPATH=$PWD ~/miniconda3/envs/torchcell/bin/python \
        experiments/026-metabolism-flux/scripts/gem_audit.py

This is the reconstruction half of the build: before any model is trained, it measures
how much of the constraint layer is data and how much is a default, and writes the answer
to ``results/gem_audit.json`` plus two figures.

It exists because the failure mode of a constrained model is silent. A capacity constraint
built on a single organism-wide turnover number is a uniform rescaling of the flux box, not
an enzyme constraint, and a thermodynamic term on reactions whose free energies are all
imputed is a regularizer wearing a physics costume. Neither shows up in a loss curve.
"""

import json
import os
import os.path as osp
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from matplotlib.colors import to_rgba
from matplotlib.ticker import MultipleLocator

from torchcell.metabolism.constraints import build_gem_tensors, compare_reaction_delta_g
from torchcell.metabolism.parameters import (
    ParameterProvenance,
    concentration_prior,
    molecular_weight_table,
    resolve_kcat_table,
)
from torchcell.metabolism.yeast_GEM import YeastGEM
from torchcell.utils import (
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    apply_paper_style,
    mm_to_in,
    savefig_true_size_svg,
)

load_dotenv()

RESULTS_DIR = osp.join(
    os.environ["EXPERIMENT_ROOT"], "026-metabolism-flux", "results"
)
IMAGES_DIR = osp.join(os.environ["ASSET_IMAGES_DIR"], "026-metabolism-flux")
OED_MIRROR = osp.join(
    os.environ["DATA_ROOT"],
    "data/enzyme_kinetics/open_enzyme_database/scerevisiae",
)


HATCH = "////"

#: Thermal energy at the growth temperature the SGA screens use, RT with
#: R = 8.314 J/mol/K and T = 303.15 K, in kJ/mol. Marked on the residual histogram as the
#: scale below which a disagreement cannot change an equilibrium constant appreciably.
RT_KJ_PER_MOL_AT_30C = 2.52


def _apply_style(ax: plt.Axes) -> None:
    """Repo figure standard: boxed axes at 0.5 pt, thin solid gridlines."""
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
    ax.tick_params(labelsize=6, width=0.5, length=2)
    # A solid pale gray rather than a black line at alpha: alpha fades whatever is drawn
    # with it, and the repo standard reserves transparency for nothing at all.
    ax.grid(True, which="both", linewidth=0.3, color="0.85")
    ax.set_axisbelow(True)


def _legend_above(ax: plt.Axes, ncol: int, handles=None, names=None) -> None:
    """Legend OUTSIDE the axes, above it. Only data is drawn inside the data area."""
    if handles is None:
        handles, names = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        names,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.01),
        ncol=ncol,
        frameon=False,
        fontsize=6,
        handlelength=1.2,
        handletextpad=0.4,
        borderpad=0.0,
        columnspacing=1.0,
    )


def coverage_figure(audit: dict[str, Any], path_stem: str) -> None:
    """Stacked bars: how much of each parameter class is measured, predicted, defaulted."""
    apply_paper_style()
    plt.rcParams.update({"hatch.linewidth": 0.4})
    rows = [
        ("Stoichiometry $S$", 1.0, 0.0, 0.0),
        (
            r"$\Delta_f G'^\circ$ (metabolite)",
            audit["thermo"]["frac_metabolites_known"],
            0.0,
            1.0 - audit["thermo"]["frac_metabolites_known"],
        ),
        (
            r"$\Delta_r G'^\circ$ (reaction)",
            audit["thermo"]["frac_reactions_known"],
            0.0,
            1.0 - audit["thermo"]["frac_reactions_known"],
        ),
        (
            "Molecular weight",
            audit["kinetics"]["mw_experimental_fraction"],
            0.0,
            1.0 - audit["kinetics"]["mw_experimental_fraction"],
        ),
        (
            r"$k_{\mathrm{cat}}$ (catalytic unit)",
            audit["kinetics"]["kcat_experimental_fraction"],
            audit["kinetics"]["kcat_predicted_fraction"],
            audit["kinetics"]["kcat_default_fraction"],
        ),
        (
            "Concentration anchor",
            audit["thermo"]["frac_metabolites_concentration_measured"],
            0.0,
            1.0 - audit["thermo"]["frac_metabolites_concentration_measured"],
        ),
    ]
    labels = [r[0] for r in rows]
    measured = np.array([r[1] for r in rows])
    predicted = np.array([r[2] for r in rows])
    default = np.array([r[3] for r in rows])

    # Full page width: the row labels are long enough that at 88.5 mm they took a third of
    # the canvas, and the figure printed at half the note's 182 mm text block.
    fig, ax = plt.subplots(
        figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(56.0))
    )
    y = np.arange(len(rows))
    # Palette order: the three levels are one ordered series, so they take slots 1, 2 and
    # the gray residual slot, and the residual carries a solid black hatch rather than a
    # paler tone, because lightness is reserved for validation against test.
    ax.barh(y, measured, color=PLOT_PALETTE[0], edgecolor="black", linewidth=0.4,
            label="measured")
    ax.barh(y, predicted, left=measured, color=PLOT_PALETTE[1], edgecolor="black",
            linewidth=0.4, label="predicted")
    ax.barh(y, default, left=measured + predicted, color=PLOT_PALETTE[5],
            edgecolor="black", linewidth=0.4, hatch=HATCH, label="organism default")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.tick_params(axis="x", which="minor", length=0)
    ax.set_xlabel("fraction of entities")
    ax.invert_yaxis()
    _legend_above(ax, ncol=3)
    _apply_style(ax)
    fig.tight_layout()
    fig.savefig(f"{path_stem}.png", dpi=300)
    savefig_true_size_svg(fig, f"{path_stem}.svg")
    plt.close(fig)


def delta_g_figure(recomputed: np.ndarray, shipped: np.ndarray, path_stem: str) -> None:
    """The two independent routes to a standard reaction energy, against each other.

    Full page width and lettered, because the figure carries two panels and a reader has
    to hold both at once: the cross in a, and where its arms land in b.
    """
    apply_paper_style()
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(62.0)),
    )
    ax = axes[0]
    # Alpha on the FACE only, via to_rgba, rather than on the artist: the cloud is dense
    # enough that opaque markers would print as one block, and no edge or hatch is drawn
    # here for the transparency to fade.
    ax.scatter(shipped, recomputed, s=1.5, color=to_rgba(PLOT_PALETTE[0], 0.4),
               edgecolors="none", label="reaction")
    lim = [min(shipped.min(), recomputed.min()), max(shipped.max(), recomputed.max())]
    ax.plot(lim, lim, color="black", linewidth=0.5, linestyle="--", label="equality")
    ax.set_xlabel(r"shipped $\Delta_r G'^\circ$ (kJ mol$^{-1}$)")
    ax.set_ylabel(r"$\sum_i S_{ij}\Delta_f G'^\circ_i$ (kJ mol$^{-1}$)")
    _legend_above(
        ax,
        ncol=2,
        handles=[
            plt.Line2D([0], [0], marker="o", markersize=2.2, linestyle="none",
                       color=PLOT_PALETTE[0]),
            plt.Line2D([0], [0], color="black", linewidth=0.5, linestyle="--"),
        ],
        names=["reaction", "equality"],
    )
    _apply_style(ax)

    ax = axes[1]
    residual = np.abs(recomputed - shipped)
    ax.hist(residual, bins=60, color=PLOT_PALETTE[1], edgecolor="black", linewidth=0.3)
    ax.set_yscale("log")
    ax.set_xlabel(r"$|$residual$|$ (kJ mol$^{-1}$)")
    ax.set_ylabel("reactions")
    ax.axvline(RT_KJ_PER_MOL_AT_30C, color="black", linewidth=0.5, linestyle=":")
    # The reference line is named in the legend ABOVE the axes and its value is in the
    # caption. It used to be an annotation parked in the panel's upper right, which is
    # exactly the text-inside-the-data-area the repo standard forbids: it collides with
    # anything that moves and it duplicates a number the caption already carries.
    _legend_above(
        ax,
        ncol=2,
        handles=[
            plt.Rectangle((0, 0), 1, 1, facecolor=PLOT_PALETTE[1], edgecolor="black",
                          linewidth=0.4),
            plt.Line2D([0], [0], color="black", linewidth=0.5, linestyle=":"),
        ],
        names=["reactions", r"$RT$ at 30 $^{\circ}$C"],
    )
    _apply_style(ax)

    for letter, ax in zip("ab", axes):
        ax.text(-0.14, 1.16, letter, transform=ax.transAxes, fontsize=8,
                fontweight="bold", va="top")

    fig.tight_layout(w_pad=2.6)
    fig.savefig(f"{path_stem}.png", dpi=300)
    savefig_true_size_svg(fig, f"{path_stem}.svg")
    plt.close(fig)


def main() -> None:
    """Measure the constraint layer's data coverage and write the audit artifacts."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)

    source = YeastGEM()
    model = source.model
    gem = build_gem_tensors(model, model_dir=source.model_dir)
    thermo = gem.thermo
    assert thermo is not None

    units = gem.catalytic_units
    mw = molecular_weight_table(source.model_dir, units.gene_ids)
    kcat = resolve_kcat_table(units, source.model_dir, OED_MIRROR)
    conc, conc_mask = concentration_prior(model, gem.met_ids, source.model_dir)

    recomputed, rec_mask = gem.standard_reaction_delta_g()
    both = rec_mask & thermo.rxn_mask

    kcat_prov = np.array([str(p) for p in kcat.provenance])
    n_units = len(kcat_prov)

    audit: dict[str, Any] = {
        "gem": {
            "model_id": gem.model_id,
            "version": source.version,
            "model_dir": source.model_dir,
            "n_metabolites": gem.n_metabolites,
            "n_reactions": gem.n_reactions,
            "n_genes": len(units.gene_ids),
            "s_nonzeros": int(gem.s._nnz()),
            "s_density": float(gem.s._nnz() / (gem.n_metabolites * gem.n_reactions)),
            "rank_s": int(gem.independent_rows.numel()),
            "nullity": int(gem.n_reactions - gem.independent_rows.numel()),
            "n_redundant_balance_rows": int(
                gem.n_metabolites - gem.independent_rows.numel()
            ),
            "n_reversible": int(gem.reversible_mask.sum()),
            "n_irreversible": int((gem.lb >= 0).sum()),
            "n_exchange": int(gem.exchange_indices.numel()),
            "biomass_reaction": gem.rxn_ids[gem.biomass_index],
            "n_reactions_with_gpr": units.n_reactions_with_gpr,
            "n_catalytic_units": units.n_units,
            "n_multigene_units": units.n_multigene_units,
            "n_nondefault_bounds": int(
                ((gem.lb != -1000) & (gem.lb != 0)).sum() + (gem.ub != 1000).sum()
            ),
        },
        "thermo": {
            "sha256": thermo.sha256,
            "source_paths": thermo.source_paths,
            "sentinel": thermo.sentinel,
            "n_metabolites_known": thermo.met_coverage.n_known,
            "frac_metabolites_known": thermo.met_coverage.fraction,
            "n_reactions_known": thermo.rxn_coverage.n_known,
            "frac_reactions_known": thermo.rxn_coverage.fraction,
            "n_reactions_all_participants_known": int(rec_mask.sum()),
            "n_metabolites_concentration_measured": int(conc_mask.sum()),
            "frac_metabolites_concentration_measured": float(conc_mask.float().mean()),
            "consistency": compare_reaction_delta_g(gem),
        },
        "kinetics": {
            "oed_mirror": OED_MIRROR,
            "n_catalytic_units": n_units,
            "kcat_experimental_fraction": float(kcat.experimental_coverage.fraction),
            "kcat_predicted_fraction": float(
                (kcat.known_mask & ~kcat.experimental_mask).float().mean()
            ),
            "kcat_default_fraction": float((~kcat.known_mask).float().mean()),
            "kcat_provenance_counts": {
                str(p): int((kcat_prov == str(p)).sum())
                for p in ParameterProvenance
                if (kcat_prov == str(p)).any()
            },
            "kcat_median_per_s": float(kcat.values.median()),
            "kcat_note": kcat.notes,
            "mw_experimental_fraction": float(mw.experimental_coverage.fraction),
            "mw_median_kda": float(mw.values.median()),
        },
    }

    with open(osp.join(RESULTS_DIR, "gem_audit.json"), "w") as f:
        json.dump(audit, f, indent=2)

    # NOT "parameter_coverage": that basename belongs to plot_parameter_coverage.py, whose
    # six-panel figure is the one the dev note and notes-tex both include. Both scripts
    # wrote it, so whichever ran last silently decided which figure the documents showed.
    coverage_figure(audit, osp.join(IMAGES_DIR, "gem_audit_parameter_coverage"))
    delta_g_figure(
        recomputed[both].numpy(),
        thermo.rxn_delta_g[both].numpy(),
        osp.join(IMAGES_DIR, "delta_g_consistency"),
    )

    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
