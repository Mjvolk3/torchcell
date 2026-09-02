# experiments/026-metabolism-flux/scripts/box_zero_reachability.py
# [[experiments.026-metabolism-flux.scripts.box_zero_reachability]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/026-metabolism-flux/scripts/box_zero_reachability.py

r"""Why a sigmoid box cannot produce a sparse, mass-balanced flux vector.

Run from the worktree root::

    PYTHONPATH=$PWD ~/miniconda3/envs/torchcell/bin/python \
        experiments/026-metabolism-flux/scripts/box_zero_reachability.py

THE OBSERVATION THIS EXPLAINS
-----------------------------
In every box-parameterized arm the mass-balance diagnostic
:math:`\operatorname{median}_i |[Sv]_i| / \omega_i` sits at **1.99** and does not move over
training, while the second-law violation fraction falls steadily. A ratio of 2.0 is not
"quite unbalanced", it is the **maximum the statistic can take**: with
:math:`\omega_i=\tfrac12\sum_j|S_{ij}v_j|`, a metabolite that is only produced and never
consumed gives :math:`|[Sv]_i| = 2\omega_i` exactly. So the median metabolite is completely
unbalanced, and no amount of penalty weight is moving it.

THE MECHANISM, AND IT IS A PROPERTY OF THE PARAMETERIZATION
-----------------------------------------------------------
.. math::
    v_j = v^{\ell}_j + (v^{u}_j - v^{\ell}_j)\,\sigma(z_j)

For the 2,463 irreversible reactions :math:`v^{\ell}_j = 0`, so :math:`v_j = 0` requires
:math:`\sigma(z_j) = 0`, i.e. :math:`z_j \to -\infty`. **Zero flux is an asymptote of the
parameterization, not a point in it.** A real flux distribution is overwhelmingly sparse,
so the balanced solutions live exactly where this parameterization cannot go, and the
optimizer's only route there is to drive thousands of logits to large negative values
against weight decay.

This script measures the gap rather than asserting it: the flux distribution the box
actually produces, how far every logit would have to move for a metabolite to balance, and
what the same statistic looks like under the null-space parameterization, where balance is
exact and the sparsity question does not arise.

The consequence is a design one. If the box is kept, it needs an explicit zero -- a gate,
a hard-concrete mask, or a shifted-sigmoid with a flat region at the lower bound -- rather
than a wider penalty weight.
"""

import json
import os
import os.path as osp

import numpy as np
import torch
from dotenv import load_dotenv

from torchcell.metabolism.constraints import (
    ThermoMode,
    build_gem_tensors,
    null_space_basis,
)
from torchcell.metabolism.flux_layer import FluxLayer, FluxLayerConfig
from torchcell.metabolism.parameters import molecular_weight_table, resolve_kcat_table
from torchcell.metabolism.yeast_GEM import YeastGEM

load_dotenv()

DATA_ROOT = os.environ["DATA_ROOT"]
RESULTS_DIR = osp.join(os.environ["EXPERIMENT_ROOT"], "026-metabolism-flux", "results")
OED_MIRROR = osp.join(
    DATA_ROOT, "data/enzyme_kinetics/open_enzyme_database/scerevisiae"
)


def _balance_stats(layer: FluxLayer, v: torch.Tensor) -> dict[str, float]:
    """Mass-balance residual as a fraction of turnover, on the independent rows."""
    residual = layer._s_matmul(v)[:, layer.independent_rows]
    omega = layer.turnover(v)[:, layer.independent_rows]
    ratio = (residual.abs() / (omega + 1e-6)).flatten()
    return {
        "median_ratio": float(ratio.median()),
        "mean_ratio": float(ratio.mean()),
        "frac_at_maximum_2": float((ratio > 1.99).float().mean()),
        "frac_below_0_1": float((ratio < 0.1).float().mean()),
        "max_abs_residual": float(residual.abs().max()),
    }


def _figure(fluxes: dict[str, np.ndarray], balances: dict[str, np.ndarray]) -> None:
    """Two panels: what magnitudes each parameterization reaches, and what it balances."""
    import matplotlib.pyplot as plt

    from torchcell.utils import (
        PANEL_WIDTHS_MM,
        PLOT_PALETTE,
        mm_to_in,
        savefig_true_size_svg,
    )

    images_dir = osp.join(os.environ["ASSET_IMAGES_DIR"], "026-metabolism-flux")
    os.makedirs(images_dir, exist_ok=True)
    plt.rcParams.update(
        {"font.family": "Arial", "font.size": 6, "svg.fonttype": "none"}
    )
    fig, axes = plt.subplots(
        1, 2, figsize=(mm_to_in(PANEL_WIDTHS_MM["half_plus"]), mm_to_in(48.0))
    )

    ax = axes[0]
    bins = np.logspace(-9, 3, 60)
    for i, (name, v) in enumerate(fluxes.items()):
        ax.hist(
            np.clip(np.abs(v), 1e-9, None),
            bins=bins,
            histtype="step",
            linewidth=0.9,
            color=PLOT_PALETTE[i],
            label=name,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    # Pin the left edge: the near-zero fluxes are the point of the panel, and autoscaling
    # crops them off screen because they sit in a single clipped bin.
    ax.set_xlim(1e-9, 1e3)
    ax.set_xlabel(r"$|v_j|$ (mmol gDW$^{-1}$ h$^{-1}$)")
    ax.set_ylabel("reactions")
    ax.legend(fontsize=6, frameon=False, loc="upper left")
    _style(ax)

    ax = axes[1]
    for i, (name, r) in enumerate(balances.items()):
        ax.hist(
            np.clip(r, 1e-10, None),
            bins=np.logspace(-10, 0.5, 60),
            histtype="step",
            linewidth=0.9,
            color=PLOT_PALETTE[i],
            label=name,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.axvline(2.0, color="black", linewidth=0.5, linestyle=":")
    ax.set_xlabel(
        "mass-balance residual per metabolite\n"
        r"$|[Sv]_i| / \omega_i$   (2 = worst possible)"
    )
    ax.set_ylabel("metabolites")
    _style(ax)

    fig.tight_layout()
    stem = osp.join(images_dir, "box_vs_nullspace")
    fig.savefig(f"{stem}.png", dpi=300)
    savefig_true_size_svg(fig, f"{stem}.svg")
    plt.close(fig)


def _style(ax) -> None:
    """Boxed axes at 0.5 pt, the repo figure standard."""
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
    ax.tick_params(labelsize=6, width=0.5, length=2)
    ax.grid(True, which="both", linewidth=0.3, alpha=0.4)
    ax.set_axisbelow(True)


def main() -> None:
    """Measure flux sparsity and balance under both parameterizations."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    torch.manual_seed(0)
    source = YeastGEM()
    gem = build_gem_tensors(source.model, model_dir=source.model_dir)
    units = gem.catalytic_units
    mw = molecular_weight_table(source.model_dir, units.gene_ids)
    kcat = resolve_kcat_table(units, source.model_dir, OED_MIRROR)
    gene_ids = units.gene_ids

    report: dict[str, object] = {
        "n_reactions": gem.n_reactions,
        "n_irreversible_lower_bound_zero": int((gem.lb >= 0).sum()),
    }

    common = dict(kcat_per_s=kcat.values, molecular_weight_kda=mw.values)
    h = torch.randn(8, len(gene_ids), 32)
    ctx = torch.randn(8, 32)
    empty = torch.zeros(0, dtype=torch.long)
    fluxes: dict[str, np.ndarray] = {}
    balances: dict[str, np.ndarray] = {}

    for name, cfg in [
        ("box", FluxLayerConfig(hidden_dim=32, thermo_mode=ThermoMode.ANCHORED)),
        (
            "nullspace",
            FluxLayerConfig(
                hidden_dim=32,
                parameterization="nullspace",
                thermo_mode=ThermoMode.ANCHORED,
            ),
        ),
    ]:
        extra = {}
        if cfg.parameterization == "nullspace":
            extra["null_space"] = null_space_basis(
                gem.s,
                cache_path=osp.join(
                    DATA_ROOT, "data/torchcell/yeast-GEM/null_space_basis_9_0_2.npy"
                ),
            )
        layer = FluxLayer(gem, gene_ids, config=cfg, **common, **extra)
        with torch.no_grad():
            out = layer(h, ctx, empty, empty)
        v = out["v"]
        absv = v.abs()
        scale = cfg.flux_scale
        fluxes[name] = v.flatten().numpy()
        residual = layer._s_matmul(v)[:, layer.independent_rows]
        omega = layer.turnover(v)[:, layer.independent_rows]
        balances[name] = (residual.abs() / (omega + 1e-6)).flatten().numpy()
        report[name] = {
            "balance": _balance_stats(layer, v),
            "flux_sparsity": {
                # A real pFBA solution has ~88 % of reactions at exactly zero. These are
                # the fractions the parameterization can actually reach.
                "frac_exactly_zero": float((absv == 0).float().mean()),
                "frac_below_1e-6_of_scale": float((absv < 1e-6 * scale).float().mean()),
                "frac_below_1e-3_of_scale": float((absv < 1e-3 * scale).float().mean()),
                "frac_below_1e-2_of_scale": float((absv < 1e-2 * scale).float().mean()),
                "median_abs_flux": float(absv.median()),
            },
            "box_violation_frac": float(out["feas_box_violation_frac"]),
        }

    # How far a logit has to travel for an irreversible reaction to reach 1e-6 of its
    # upper bound. This is the number that makes "asymptote" concrete.
    for target in (1e-3, 1e-6, 1e-9):
        report[f"logit_for_sigma_{target:g}"] = float(np.log(target / (1 - target)))

    with open(osp.join(RESULTS_DIR, "box_zero_reachability.json"), "w") as f:
        json.dump(report, f, indent=2)
    _figure(fluxes, balances)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
