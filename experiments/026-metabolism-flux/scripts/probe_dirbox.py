# experiments/026-metabolism-flux/scripts/probe_dirbox.py
# [[experiments.026-metabolism-flux.physics-gates-2026-09-14]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/026-metabolism-flux/scripts/probe_dirbox
r"""Is the standard-state directional polytope anything other than the origin?

WHAT IS ASKED. ``FluxLayer._thermo_tighten`` turns the second law into a box: a reaction
whose standard reaction energy is positive may not run forward, one whose energy is
negative may not run backward. Applying that rule at STANDARD state to every priced,
non-exempt reaction and then asking cobra two questions,

* what is the maximum growth inside the tightened box, and
* what is the maximum total downhill throughput inside it,

says whether the set the ``tdg`` head projects into contains a working cell at all. Every
term of the throughput objective is non-negative under the sign restriction, so a maximum
of exactly zero means every priced non-exempt reaction is zero at every feasible point,
i.e. the tightened polytope is exactly ``{v = 0}`` on the priced set.

That is the geometric form of the ``tdg`` result ("drove flux to ~zero with zero signal"),
derived from the box rather than observed after eighty epochs of training.

The margin is swept, because ``thermo_box_margin_kj`` exists precisely to leave
near-equilibrium reactions unpinned and the question is whether a margin rescues the set.

Run from the repo root::

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=$PWD ~/miniconda3/envs/torchcell/bin/python \
        experiments/026-metabolism-flux/scripts/probe_dirbox.py

Writes ``experiments/026-metabolism-flux/results/probe_dirbox.json``.
"""

from __future__ import annotations

import json
import os
import os.path as osp
import time

import numpy as np
from dotenv import load_dotenv

load_dotenv()

from torchcell.metabolism.constraints import build_gem_tensors  # noqa: E402
from torchcell.metabolism.yeast_GEM import YeastGEM  # noqa: E402

EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "026-metabolism-flux", "results")

#: Margins, in kJ/mol, at which the directional box is built. 1e-3 is the layer's
#: `thermo_epsilon`, i.e. what the first tdg cells ran; the other two are `tdgm5` and a
#: margin wide enough that only strongly driven reactions are pinned.
MARGINS_KJ: tuple[float, ...] = (1.0e-3, 5.0, 20.0)
#: A reaction counts as carrying flux above this, mmol/gDW/h.
FLUX_ACTIVE = 1.0e-4


def main() -> None:
    started = time.time()
    source = YeastGEM()
    model = source.model
    gem = build_gem_tensors(
        model, source.model_dir, with_thermo=True, with_independent_rows=False
    )
    delta_g0_t, mask_t = gem.standard_reaction_delta_g()
    delta_g0 = delta_g0_t.double().numpy()
    priced = mask_t.numpy()
    rxn_ids = gem.rxn_ids
    exchanges = set(gem.exchange_indices.tolist())
    biomass = int(gem.biomass_index)
    print(
        f"priced {int(priced.sum())} of {len(rxn_ids)}; exchanges {len(exchanges)}; "
        f"biomass {rxn_ids[biomass]}; loaded in {time.time() - started:.1f}s"
    )

    base_lb = np.array([r.lower_bound for r in model.reactions])
    base_ub = np.array([r.upper_bound for r in model.reactions])
    # Thermo-Flux Box 2's exemptions, the same set the layer builds: exchanges and biomass.
    constrained = priced.copy()
    for j in exchanges:
        constrained[j] = False
    constrained[biomass] = False

    with model:
        unconstrained = model.optimize()
    unconstrained_growth = (
        float(unconstrained.objective_value)
        if unconstrained.status == "optimal"
        else None
    )
    print(f"unconstrained growth: {unconstrained_growth}")

    rows: list[dict[str, object]] = []
    for margin in MARGINS_KJ:
        uphill = constrained & (delta_g0 > margin)
        downhill = constrained & (delta_g0 < -margin)
        lo, hi = base_lb.copy(), base_ub.copy()
        hi[uphill] = np.minimum(hi[uphill], 0.0)
        lo[downhill] = np.maximum(lo[downhill], 0.0)
        emptied = int((lo > hi).sum())
        lo = np.minimum(lo, hi)
        with model:
            for j, reaction in enumerate(model.reactions):
                reaction.lower_bound = float(lo[j])
                reaction.upper_bound = float(hi[j])
            growth_solution = model.optimize()
            growth = (
                float(growth_solution.objective_value)
                if growth_solution.status == "optimal"
                else None
            )
            growth_active = None
            if growth_solution.status == "optimal":
                v = np.array([growth_solution.fluxes[r] for r in rxn_ids])
                growth_active = int((np.abs(v) > FLUX_ACTIVE).sum())
            # The second LP drops growth entirely and maximizes total DOWNHILL throughput,
            # which under the sign restriction is a sum of non-negative terms.
            weights = np.zeros(len(rxn_ids))
            weights[uphill] = -1.0
            weights[downhill] = +1.0
            model.objective = {
                model.reactions[j]: float(weights[j])
                for j in range(len(weights))
                if weights[j] != 0.0
            }
            throughput_solution = model.optimize()
            throughput = (
                float(throughput_solution.objective_value)
                if throughput_solution.status == "optimal"
                else None
            )
            throughput_active = None
            if throughput_solution.status == "optimal":
                v2 = np.array([throughput_solution.fluxes[r] for r in rxn_ids])
                throughput_active = int((np.abs(v2) > FLUX_ACTIVE).sum())
        rows.append(
            {
                "margin_kj": margin,
                "n_uphill_pinned": int(uphill.sum()),
                "n_downhill_pinned": int(downhill.sum()),
                "emptied_boxes": emptied,
                "max_growth": growth,
                "growth_status": growth_solution.status,
                "growth_rxn_active": growth_active,
                "max_downhill_throughput": throughput,
                "throughput_status": throughput_solution.status,
                "throughput_rxn_active": throughput_active,
            }
        )
        print(json.dumps(rows[-1], indent=2))

    payload = {
        "definitions": {
            "max_downhill_throughput": (
                "the LP maximum of sum_j sign-licensed v_j over the tightened box. Every "
                "term is non-negative there, so a maximum of exactly 0 means every priced "
                "non-exempt reaction is zero at every feasible point."
            )
        },
        "unconstrained_growth": unconstrained_growth,
        "n_reactions": len(rxn_ids),
        "n_priced": int(priced.sum()),
        "n_constrained_after_box2_exemptions": int(constrained.sum()),
        "margins": rows,
    }
    out_path = osp.join(RESULTS_DIR, "probe_dirbox.json")
    with open(out_path, "w") as handle:
        json.dump(payload, handle, indent=2)
    print(f"wrote {out_path} in {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
