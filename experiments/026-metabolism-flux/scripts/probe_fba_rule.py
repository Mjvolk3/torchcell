# experiments/026-metabolism-flux/scripts/probe_fba_rule.py
# [[experiments.026-metabolism-flux.physics-gates-2026-09-14]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/026-metabolism-flux/scripts/probe_fba_rule
r"""Calibrate the feasibility audit's balance rules against a REAL flux on this GEM.

WHY THIS EXISTS. Four waves of 027 cells were scored against a balance rule whose target
had never been measured. The rule is a statistic over the independent rows of S, and a
statistic has no meaning until something known-good has been put through it. This script
puts three known-good fluxes through the exact statistic
``experiments/027-betaxanthin-metabolic/scripts/audit_top_strain_feasibility.py`` uses
(``flux_scale`` 100, turnover threshold 1e-3 x scale, ``balance_epsilon`` 1e-6, the
independent rows of ``build_gem_tensors``):

* the FBA solution at maximum biomass,
* the pFBA solution, which is the sparsest flux reaching that biomass,
* a random-objective LP vertex, which is a generic corner of the same polytope.

TWO THINGS IT SETTLES, and both are recorded in the JSON rather than asserted here.

**The max-over-ALL-rows rule is mis-specified.** A real FBA solution fails it. The failing
rows are rows whose turnover sits at the ``balance_epsilon`` floor, where the denominator
is 1e-6 and the numerator is LP solver residue, so the ratio is decided by idle metabolites
and not by the flux. That rule is a solver-noise diagnostic and is reported as one.

**The turnover-thresholded rule is reachable and it has a scale.** The reference numbers it
produces (the active-row count, the active-row max and p95, and the fraction of reactions
carrying flux) are what any head's numbers have to be read against.

Run from the repo root::

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=$PWD ~/miniconda3/envs/torchcell/bin/python \
        experiments/026-metabolism-flux/scripts/probe_fba_rule.py

Writes ``experiments/026-metabolism-flux/results/probe_fba_rule.json``.
"""

from __future__ import annotations

import json
import os
import os.path as osp
import time

import numpy as np
from cobra.flux_analysis import pfba
from dotenv import load_dotenv

load_dotenv()

from torchcell.metabolism.constraints import build_gem_tensors  # noqa: E402
from torchcell.metabolism.yeast_GEM import YeastGEM  # noqa: E402

EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "026-metabolism-flux", "results")

#: The layer's working flux magnitude, ``FluxLayerConfig.flux_scale``.
FLUX_SCALE = 100.0
#: Turnover a row must clear to be inside the thresholded rule, as a fraction of the scale.
TURNOVER_THRESHOLD = 1.0e-3
#: ``FluxLayerConfig.balance_epsilon``, the floor on the ratio's denominator.
BALANCE_EPSILON = 1.0e-6
#: A reaction carries flux above this fraction of the scale, matching the audit's
#: ``FLUX_ACTIVE_RELATIVE_THRESHOLD``.
FLUX_ACTIVE_RELATIVE_THRESHOLD = 1.0e-6


def statistics(
    s: np.ndarray, rows: np.ndarray, v: np.ndarray, label: str, biomass: int | None
) -> dict[str, object]:
    """Every balance statistic the audit reports, on one flux vector."""
    residual = np.abs(s[rows] @ v)
    omega = 0.5 * np.abs(s[rows]) @ np.abs(v)
    ratio = residual / (omega + BALANCE_EPSILON)
    active = omega > TURNOVER_THRESHOLD * FLUX_SCALE
    out: dict[str, object] = {
        "label": label,
        "flux_active_frac": float(
            (np.abs(v) > FLUX_ACTIVE_RELATIVE_THRESHOLD * FLUX_SCALE).mean()
        ),
        "n_active_rows": int(active.sum()),
        "balance_max_all": float(ratio.max()),
        "balance_median_all": float(np.median(ratio)),
        "balance_active_max": float(ratio[active].max()) if active.any() else None,
        "balance_active_p95": (
            float(np.percentile(ratio[active], 95)) if active.any() else None
        ),
        "abs_residual_max": float(residual.max()),
        "biomass_flux": None if biomass is None else float(v[biomass]),
    }
    print(json.dumps(out, indent=2))
    return out


def main() -> None:
    started = time.time()
    source = YeastGEM()
    model = source.model
    gem = build_gem_tensors(
        model, source.model_dir, with_thermo=False, with_independent_rows=True
    )
    s = gem.s.to_dense().double().numpy()
    rows = gem.independent_rows.numpy()
    biomass = None if gem.biomass_index is None else int(gem.biomass_index)
    print(f"loaded in {time.time() - started:.1f}s  S={s.shape} independent rows={len(rows)}")

    rows_out: list[dict[str, object]] = []
    solution = model.optimize()
    assert solution.status == "optimal", f"the base FBA is {solution.status}."
    v = np.array([solution.fluxes[r] for r in gem.rxn_ids])
    rows_out.append(statistics(s, rows, v, "FBA (max biomass)", biomass))

    parsimonious = pfba(model)
    v_pfba = np.array([parsimonious.fluxes[r] for r in gem.rxn_ids])
    rows_out.append(statistics(s, rows, v_pfba, "pFBA", biomass))

    rng = np.random.default_rng(0)
    with model:
        weights = rng.normal(size=len(gem.rxn_ids))
        model.objective = {
            model.reactions[j]: float(weights[j]) for j in range(len(weights))
        }
        vertex = model.optimize()
        assert vertex.status == "optimal", f"the random-objective LP is {vertex.status}."
        v_vertex = np.array([vertex.fluxes[r] for r in gem.rxn_ids])
    rows_out.append(statistics(s, rows, v_vertex, "random-objective LP vertex", biomass))

    payload = {
        "definitions": {
            "balance_max_all": (
                "max over the independent rows of |(Sv)_i| / (omega_i + 1e-6). A row whose "
                "turnover sits at the epsilon floor reads 2 whenever its two tiny fluxes "
                "share a sign, so this statistic is decided by idle metabolites and by LP "
                "solver residue. SOLVER-NOISE DIAGNOSTIC, not a feasibility rule."
            ),
            "balance_active_max": (
                "the same ratio over the rows whose turnover exceeds 1e-3 * flux_scale = "
                "0.1 mmol/gDW/h. This is the rule the audit calls Genesis."
            ),
            "flux_active_frac": (
                "fraction of reactions with |v| > 1e-6 * flux_scale = 1e-4 mmol/gDW/h."
            ),
        },
        "flux_scale": FLUX_SCALE,
        "turnover_threshold": TURNOVER_THRESHOLD,
        "balance_epsilon": BALANCE_EPSILON,
        "n_reactions": int(s.shape[1]),
        "n_metabolites": int(s.shape[0]),
        "n_independent_rows": int(len(rows)),
        "fluxes": rows_out,
    }
    out_path = osp.join(RESULTS_DIR, "probe_fba_rule.json")
    with open(out_path, "w") as handle:
        json.dump(payload, handle, indent=2)
    print(f"wrote {out_path} in {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
