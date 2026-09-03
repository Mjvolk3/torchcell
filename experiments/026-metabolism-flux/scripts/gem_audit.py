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


def _apply_style(ax: plt.Axes) -> None:
    """Repo figure standard: boxed axes, 6 pt Arial, tenth gridlines."""
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
    ax.tick_params(labelsize=6, width=0.5, length=2)
    ax.grid(True, which="both", linewidth=0.3, alpha=0.4)
    ax.set_axisbelow(True)


def coverage_figure(audit: dict[str, Any], path_stem: str) -> None:
    """Stacked bars: how much of each parameter class is measured, predicted, defaulted."""
    plt.rcParams.update({"font.family": "Arial", "font.size": 6, "svg.fonttype": "none"})
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

    fig, ax = plt.subplots(
        figsize=(mm_to_in(PANEL_WIDTHS_MM["half_plus"]), mm_to_in(52.0))
    )
    y = np.arange(len(rows))
    ax.barh(y, measured, color=PLOT_PALETTE[4], edgecolor="black", linewidth=0.4,
            label="measured")
    ax.barh(y, predicted, left=measured, color=PLOT_PALETTE[0], edgecolor="black",
            linewidth=0.4, label="predicted")
    ax.barh(y, default, left=measured + predicted, color=PLOT_PALETTE[5],
            edgecolor="black", linewidth=0.4, hatch="///", label="organism default")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.tick_params(which="minor", length=0)
    ax.set_xlabel("fraction of entities")
    ax.invert_yaxis()
    # Below the axis, not inside it: the kcat and concentration rows are almost entirely
    # "organism default", so an in-axes legend lands on top of the two bars the figure
    # exists to show.
    ax.legend(
        fontsize=6,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.28),
        ncol=3,
        handlelength=1.4,
        columnspacing=1.2,
    )
    _apply_style(ax)
    fig.tight_layout()
    fig.savefig(f"{path_stem}.png", dpi=300)
    savefig_true_size_svg(fig, f"{path_stem}.svg")
    plt.close(fig)


def delta_g_figure(recomputed: np.ndarray, shipped: np.ndarray, path_stem: str) -> None:
    """The two independent routes to a standard reaction energy, against each other."""
    plt.rcParams.update({"font.family": "Arial", "font.size": 6, "svg.fonttype": "none"})
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(mm_to_in(PANEL_WIDTHS_MM["half_plus"]), mm_to_in(48.0)),
    )
    ax = axes[0]
    ax.scatter(shipped, recomputed, s=1.5, color=PLOT_PALETTE[4], alpha=0.4,
               edgecolors="none")
    lim = [min(shipped.min(), recomputed.min()), max(shipped.max(), recomputed.max())]
    ax.plot(lim, lim, color="black", linewidth=0.5, linestyle="--")
    ax.set_xlabel(r"shipped $\Delta_r G'^\circ$ (kJ mol$^{-1}$)")
    ax.set_ylabel(r"$\sum_i S_{ij}\Delta_f G'^\circ_i$")
    _apply_style(ax)

    ax = axes[1]
    residual = np.abs(recomputed - shipped)
    ax.hist(residual, bins=60, color=PLOT_PALETTE[1], edgecolor="black", linewidth=0.3)
    ax.set_yscale("log")
    ax.set_xlabel(r"$|$residual$|$ (kJ mol$^{-1}$)")
    ax.set_ylabel("reactions")
    ax.axvline(2.52, color="black", linewidth=0.5, linestyle=":")
    # Annotated in AXES coordinates, horizontally, in the panel's empty upper right.
    # The previous version placed it in DATA coordinates on a LOG axis and rotated it
    # 90 degrees, so `ylim[1] * 0.4` put its baseline near the top of the axes and the
    # rotated string then ran off the panel and was clipped mid-word. A log axis makes
    # a fraction of the upper limit a position near the top, not near the middle.
    ax.text(
        0.97,
        0.95,
        r"dotted: $RT$ at $30^{\circ}$C",
        transform=ax.transAxes,
        fontsize=6,
        ha="right",
        va="top",
    )
    _apply_style(ax)

    fig.tight_layout()
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

    coverage_figure(audit, osp.join(IMAGES_DIR, "parameter_coverage"))
    delta_g_figure(
        recomputed[both].numpy(),
        thermo.rxn_delta_g[both].numpy(),
        osp.join(IMAGES_DIR, "delta_g_consistency"),
    )

    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
