# experiments/026-metabolism-flux/scripts/probe_tfa_lp.py
# [[experiments.026-metabolism-flux.physics-gates-2026-09-14]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/026-metabolism-flux/scripts/probe_tfa_lp
r"""Can ANY concentration vector in the window make a real growth flux sign-consistent?

THE LP. This is the standard thermodynamic-feasibility program with the flux held fixed at
a cobra solution. For every flux-carrying, priced, non-exempt reaction :math:`j`, the
second law asks :math:`\operatorname{sign}\Delta_rG_j=-\operatorname{sign}v_j`, and with
:math:`\Delta_rG_j=\Delta_rG^\circ_j+RT(S^\top\ln c)_j` that is one linear inequality in
:math:`\ln c`. Bounding :math:`\ln c` to a concentration window and minimizing the total
L1 slack turns "is there such a concentration" into an LP that always solves: a total slack
of zero is feasibility, and a positive slack names, in kJ/mol, how far outside physics the
table puts the flux.

WHAT THE SLACK IS AND IS NOT. The LP minimizes the TOTAL slack, so the identity of the
violated reactions is the LP's choice of where to place a JOINT infeasibility. It is not a
per-reaction verdict, and the JSON says so beside every offender list.

ROUTES AND WINDOWS. ``--route`` selects one of the five thermodynamic routes or ``all``;
each is evaluated in the physiological window (1e-7 to 1e-2 M, Thermo-Flux's default and
the layer's ``conc_lower_m``/``conc_upper_m``) and in a deliberately absurd one (1e-12 to
10 M), because a violation that survives the absurd window is a wrong formation energy
rather than a tight window. ``--route shipped`` additionally runs the epsilon sweep and the
random-vertex control.

This script merges three probes of the 2026-09-14 physics review: the route sweep, the
window sweep, and the named-offender listing.

Run from the repo root::

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=$PWD ~/miniconda3/envs/torchcell/bin/python \
        experiments/026-metabolism-flux/scripts/probe_tfa_lp.py --route all

Writes ``experiments/026-metabolism-flux/results/probe_tfa_lp.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import time

import numpy as np
from dotenv import load_dotenv
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, eye, hstack

load_dotenv()

from torchcell.metabolism.constraints import (  # noqa: E402
    CONC_LOWER_M,
    CONC_UPPER_M,
    build_gem_tensors,
)
from torchcell.metabolism.parameters import (  # noqa: E402
    ThermoSource,
    resolve_reaction_delta_g,
)
from torchcell.metabolism.yeast_GEM import YeastGEM  # noqa: E402

DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "026-metabolism-flux", "results")
THERMO_DIR = osp.join(DATA_ROOT, "data/torchcell/thermo_equilibrator")

#: RT in kJ/mol at the compartment table's 303.15 K, the value the layer uses.
RT_KJ_PER_MOL = 8.314462618e-3 * 303.15
#: A reaction carries flux above this, mmol/gDW/h.
FLUX_ACTIVE = 1.0e-6
#: A slack counts as a violation above this, kJ/mol.
SLACK_TOLERANCE = 1.0e-6
#: The two windows, in molar. The first is the layer's shipped window.
WINDOWS: tuple[tuple[float, float], ...] = ((CONC_LOWER_M, CONC_UPPER_M), (1e-12, 10.0))
ROUTES: tuple[ThermoSource, ...] = (
    ThermoSource.SHIPPED,
    ThermoSource.EQUILIBRATOR,
    ThermoSource.UNION,
    ThermoSource.UNION_EXTERNAL,
    ThermoSource.VERIFIED,
)


def feasibility_lp(
    s: np.ndarray,
    delta_g0: np.ndarray,
    constrained: np.ndarray,
    v: np.ndarray,
    window_m: tuple[float, float],
    epsilon_kj: float,
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    """Minimum total L1 slack on the sign constraints. Returns the row, indices, slacks."""
    lo, hi = np.log(window_m[0]), np.log(window_m[1])
    carrying = (np.abs(v) > FLUX_ACTIVE) & constrained
    idx = np.flatnonzero(carrying)
    # v > 0 asks dG <= -eps, i.e.  RT S^T ln c <= -eps - dG0
    # v < 0 asks dG >= +eps, i.e. -RT S^T ln c <=  dG0 - eps
    rows = []
    rhs = []
    for j in idx:
        column = s[:, j]
        if v[j] > 0:
            rows.append(RT_KJ_PER_MOL * column)
            rhs.append(-epsilon_kj - delta_g0[j])
        else:
            rows.append(-RT_KJ_PER_MOL * column)
            rhs.append(delta_g0[j] - epsilon_kj)
    a_ub = csr_matrix(np.array(rows))
    b_ub = np.array(rhs)
    n_met, n_con = s.shape[0], len(b_ub)
    program = linprog(
        np.concatenate([np.zeros(n_met), np.ones(n_con)]),
        A_ub=hstack([a_ub, -eye(n_con, format="csr")], format="csr"),
        b_ub=b_ub,
        bounds=[(lo, hi)] * n_met + [(0, None)] * n_con,
        method="highs",
    )
    assert program.success, f"the slack LP did not solve: {program.message}"
    slack = program.x[n_met:]
    violated = slack > SLACK_TOLERANCE
    row = {
        "window_m": list(window_m),
        "epsilon_kj": epsilon_kj,
        "n_constrained_reactions": int(n_con),
        "n_violated": int(violated.sum()),
        "total_slack_kj": float(program.fun),
        "max_violation_kj": float(slack.max()) if n_con else 0.0,
    }
    return row, idx, slack


def offenders(
    model, rxn_ids: list[str], delta_g0: np.ndarray, v: np.ndarray,
    idx: np.ndarray, slack: np.ndarray, limit: int = 25
) -> list[dict[str, object]]:
    """The worst reactions by slack, with the metadata that says what kind of reaction."""
    order = np.argsort(-slack)
    out: list[dict[str, object]] = []
    for k in order[:limit]:
        if slack[k] <= SLACK_TOLERANCE:
            break
        j = int(idx[k])
        reaction = model.reactions[j]
        out.append(
            {
                "reaction": rxn_ids[j],
                "name": reaction.name,
                "slack_kj": float(slack[k]),
                "delta_r_g0_kj": float(delta_g0[j]),
                "v": float(v[j]),
                "n_metabolites": len(reaction.metabolites),
                "compartments": sorted({m.compartment for m in reaction.metabolites}),
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--route",
        default="all",
        choices=[*(s.value for s in ROUTES), "all"],
        help="Which thermodynamic route to price the reactions with.",
    )
    parser.add_argument(
        "--out", default="probe_tfa_lp.json", help="File name under results/."
    )
    args = parser.parse_args()

    started = time.time()
    source = YeastGEM()
    model = source.model
    gem = build_gem_tensors(
        model, source.model_dir, with_thermo=True, with_independent_rows=False
    )
    s = gem.s.to_dense().double().numpy()
    rxn_ids = gem.rxn_ids
    exchanges = set(gem.exchange_indices.tolist())
    biomass = int(gem.biomass_index)
    solution = model.optimize()
    assert solution.status == "optimal", f"the base FBA is {solution.status}."
    v = np.array([solution.fluxes[r] for r in rxn_ids])
    print(
        f"window {CONC_LOWER_M:g}..{CONC_UPPER_M:g} M; RT {RT_KJ_PER_MOL:.4f} kJ/mol; "
        f"loaded in {time.time() - started:.1f}s"
    )

    def box2(mask: np.ndarray) -> np.ndarray:
        constrained = mask.copy()
        for j in exchanges:
            constrained[j] = False
        constrained[biomass] = False
        return constrained

    wanted = ROUTES if args.route == "all" else (ThermoSource(args.route),)
    route_rows: list[dict[str, object]] = []
    for route in wanted:
        if route is ThermoSource.SHIPPED:
            delta_g0_t, mask_t = gem.standard_reaction_delta_g()
            delta_g0 = delta_g0_t.double().numpy()
            mask = mask_t.numpy()
        else:
            table = resolve_reaction_delta_g(gem, THERMO_DIR, source=route)
            delta_g0 = np.asarray(table.delta_r_g0, dtype=float)
            mask = np.asarray(table.mask, dtype=bool)
        constrained = box2(mask)
        print(f"-- {route.value}: priced {int(mask.sum())} of {len(rxn_ids)}")
        for window in WINDOWS:
            row, idx, slack = feasibility_lp(
                s, delta_g0, constrained, v, window, epsilon_kj=1.0e-3
            )
            row["route"] = route.value
            row["flux"] = "FBA max biomass"
            row["offenders"] = offenders(model, rxn_ids, delta_g0, v, idx, slack)
            route_rows.append(row)
            print(
                f"{route.value:16s} window {window[0]:g}..{window[1]:g}  "
                f"constrained {row['n_constrained_reactions']:4d}  "
                f"violated {row['n_violated']:3d}  "
                f"L1 {row['total_slack_kj']:10.1f} kJ  max {row['max_violation_kj']:8.2f}"
            )

    sensitivity: list[dict[str, object]] = []
    if args.route in {"all", ThermoSource.SHIPPED.value}:
        # The epsilon sweep and the random-vertex control, on the shipped route only: they
        # answer whether the infeasibility is an artifact of the strictness of the
        # inequality or of the particular flux, and neither depends on the route.
        delta_g0_t, mask_t = gem.standard_reaction_delta_g()
        delta_g0 = delta_g0_t.double().numpy()
        constrained = box2(mask_t.numpy())
        for epsilon in (0.0, 1.0e-3, 1.0):
            row, _, _ = feasibility_lp(
                s, delta_g0, constrained, v, WINDOWS[0], epsilon_kj=epsilon
            )
            row["route"] = ThermoSource.SHIPPED.value
            row["flux"] = "FBA max biomass"
            sensitivity.append(row)
            print(f"eps sweep {json.dumps(row)}")
        rng = np.random.default_rng(0)
        with model:
            weights = rng.normal(size=len(rxn_ids))
            model.objective = {
                model.reactions[j]: float(weights[j]) for j in range(len(weights))
            }
            vertex = model.optimize()
            assert vertex.status == "optimal", f"random LP is {vertex.status}."
            v_vertex = np.array([vertex.fluxes[r] for r in rxn_ids])
        row, _, _ = feasibility_lp(
            s, delta_g0, constrained, v_vertex, WINDOWS[0], epsilon_kj=1.0e-3
        )
        row["route"] = ThermoSource.SHIPPED.value
        row["flux"] = "random-objective LP vertex"
        sensitivity.append(row)
        print(f"random vertex {json.dumps(row)}")

    payload = {
        "definitions": {
            "total_slack_kj": (
                "the LP minimum of the summed L1 violation of the sign constraints over "
                "ln c in the window. Zero is thermodynamic feasibility of the fixed flux."
            ),
            "offenders": (
                "the reactions carrying slack at the LP optimum. The LP minimizes the TOTAL "
                "slack, so this is where it chose to place a JOINT infeasibility, not a "
                "per-reaction verdict."
            ),
        },
        "rt_kj_per_mol": RT_KJ_PER_MOL,
        "flux_active_threshold": FLUX_ACTIVE,
        "routes": route_rows,
        "sensitivity": sensitivity,
    }
    out_path = osp.join(RESULTS_DIR, args.out)
    with open(out_path, "w") as handle:
        json.dump(payload, handle, indent=2)
    print(f"wrote {out_path} in {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
