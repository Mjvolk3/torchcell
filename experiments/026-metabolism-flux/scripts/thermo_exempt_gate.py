# experiments/026-metabolism-flux/scripts/thermo_exempt_gate.py
# [[experiments.026-metabolism-flux.physics-gates-2026-09-14]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/026-metabolism-flux/scripts/thermo_exempt_gate
r"""Is the thermodynamic infeasibility CONCENTRATED in a few reactions, or spread out?

THE GATE. ``probe_tfa_lp.py`` measures that no concentration vector in the physiological
window makes a real growth flux sign-consistent under any of the five thermodynamic routes,
with a total L1 slack of thousands of kJ/mol. The proposed remedy (option B of the
2026-09-14 review) is to exempt the reactions a table cannot defend, on the same argument
``thermo_box_margin_kj`` already makes for near-equilibrium reactions and with yETFL's
precedent of exempting all 1,304 of its membrane reactions.

That remedy is only worth a GPU if the infeasibility is concentrated. This script tests it
on CPU, in minutes, before any cell is queued:

1. solve the slack LP once and take the reactions carrying slack at its optimum,
2. drop exactly those from the constraint set and re-solve,
3. read the residual total slack against the ~10 kJ/mol gate.

**Why step 2 can fail even when step 1 names 47 reactions.** The LP minimizes the TOTAL
slack, so its choice of where to put a JOINT infeasibility is not a minimal removal set: a
different 47 may be left over once the first 47 are gone. When that happens the script
falls through to a greedy minimum-cardinality removal, dropping the single worst reaction
and re-solving until the gate is met, and records which method produced the list.

KILL RULE, stated in the review and repeated here because the JSON is what a later reader
will have: if no removal of a small set drops the L1 slack below about 10 kJ, the
infeasibility is distributed rather than concentrated, exempting reactions is not the lever,
and E2 is dead before any GPU runs.

Run from the repo root::

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=$PWD ~/miniconda3/envs/torchcell/bin/python \
        experiments/026-metabolism-flux/scripts/thermo_exempt_gate.py --route union_external

Writes ``experiments/026-metabolism-flux/results/thermo_exempt_candidates.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import os.path as osp
import time

import numpy as np
from dotenv import load_dotenv
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, eye, hstack, vstack

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

RT_KJ_PER_MOL = 8.314462618e-3 * 303.15
FLUX_ACTIVE = 1.0e-6
SLACK_TOLERANCE = 1.0e-6
#: The gate, in kJ/mol of TOTAL L1 slack. Below this the flux is thermodynamically
#: consistent to within rounding and E2 has a premise.
GATE_KJ = 10.0
#: A cap on the greedy rounds, so a distributed infeasibility terminates with an answer
#: rather than removing a tenth of the network one reaction at a time.
GREEDY_MAX_REMOVALS = 200


def solve(
    s: np.ndarray,
    delta_g0: np.ndarray,
    v: np.ndarray,
    idx: np.ndarray,
    window_m: tuple[float, float],
    epsilon_kj: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    r"""Minimum total L1 slack over ``ln c`` for the sign constraints on ``idx``.

    Returns ``(total_slack, per_constraint_slack, ln_c)``. The third is the CONCENTRATION
    WITNESS, in the GEM's own metabolite order. When the total slack is zero it is a vector
    under which the fixed flux runs downhill everywhere it is priced, which is exactly the
    concentration the flux layer's directional box has to be evaluated at if that box is to
    license that flux rather than forbid it.
    """
    lo, hi = np.log(window_m[0]), np.log(window_m[1])
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
    if not rows:
        return 0.0, np.zeros(0), np.full(s.shape[0], np.log(window_m[0]))
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
    return float(program.fun), program.x[n_met:], program.x[:n_met]


def centered_witness(
    s: np.ndarray,
    delta_g0: np.ndarray,
    v: np.ndarray,
    idx: np.ndarray,
    window_m: tuple[float, float],
    epsilon_kj: float,
    slack_budget: float,
) -> tuple[np.ndarray, float, float]:
    r"""The same feasible set, but the point closest to the middle of the window.

    WHY THE PLAIN WITNESS IS A BAD INITIALIZATION. :func:`solve` minimizes total slack and
    nothing else, so it returns an LP VERTEX: every metabolite no constraint touches is
    pushed to whichever window bound the simplex reached first, and on this GEM 2,604 of
    2,806 metabolites come back at the 10 mM ceiling. That vector licenses the flux's
    directions and is not a concentration profile, and as an initialization it starts the
    learned concentration flat against a bound where the squashing sigmoid has no gradient
    left.

    This is the lexicographic second stage: hold the total slack at what the first stage
    achieved, then minimize :math:`\sum_i|\ln c_i-\text{mid}|` over the same set. Every
    metabolite the constraints do not pin lands exactly at mid-window, and the ones that are
    pinned move only as far as they must.

    Returns ``(ln_c, total_slack, mean_absolute_distance_from_mid)``.
    """
    lo, hi = np.log(window_m[0]), np.log(window_m[1])
    mid = 0.5 * (lo + hi)
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
    a_sign = csr_matrix(np.array(rows))
    b_sign = np.array(rhs)
    n_met, n_con = s.shape[0], len(b_sign)
    # Variables: [ln_c (n_met), slack (n_con), t (n_met)].
    zero_met = csr_matrix((n_con, n_met))
    sign_block = hstack(
        [a_sign, -eye(n_con, format="csr"), zero_met], format="csr"
    )
    identity = eye(n_met, format="csr")
    zero_con = csr_matrix((n_met, n_con))
    # ln_c - t <= mid and -ln_c - t <= -mid together give t >= |ln_c - mid|.
    upper = hstack([identity, zero_con, -identity], format="csr")
    lower = hstack([-identity, zero_con, -identity], format="csr")
    # The slack budget, so feasibility is not traded for centering.
    budget = csr_matrix(
        np.concatenate([np.zeros(n_met), np.ones(n_con), np.zeros(n_met)]).reshape(1, -1)
    )
    a_ub = vstack([sign_block, upper, lower, budget], format="csr")
    b_ub = np.concatenate(
        [
            b_sign,
            np.full(n_met, mid),
            np.full(n_met, -mid),
            np.array([slack_budget + SLACK_TOLERANCE]),
        ]
    )
    cost = np.concatenate([np.zeros(n_met), np.zeros(n_con), np.ones(n_met)])
    program = linprog(
        cost,
        A_ub=a_ub,
        b_ub=b_ub,
        bounds=[(lo, hi)] * n_met + [(0, None)] * n_con + [(0, None)] * n_met,
        method="highs",
    )
    assert program.success, f"the centering LP did not solve: {program.message}"
    ln_c = program.x[:n_met]
    slack = float(program.x[n_met : n_met + n_con].sum())
    return ln_c, slack, float(np.abs(ln_c - mid).mean())


def describe(model, rxn_ids: list[str], delta_g0: np.ndarray, v: np.ndarray, j: int):
    reaction = model.reactions[j]
    return {
        "reaction": rxn_ids[j],
        "name": reaction.name,
        "delta_r_g0_kj": float(delta_g0[j]),
        "v": float(v[j]),
        "compartments": sorted({m.compartment for m in reaction.metabolites}),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--route",
        default=ThermoSource.UNION_EXTERNAL.value,
        choices=[s.value for s in ThermoSource],
        help="Which thermodynamic route to price with. union_external is what wave 7/8 ran.",
    )
    parser.add_argument("--epsilon-kj", type=float, default=1.0e-3)
    parser.add_argument("--gate-kj", type=float, default=GATE_KJ)
    parser.add_argument("--out", default="thermo_exempt_candidates.json")
    parser.add_argument(
        "--witness-out",
        default="thermo_consistent_log_c.json",
        help="Where the LP's concentration witness is written, under results/.",
    )
    parser.add_argument(
        "--centered-out",
        default="thermo_consistent_log_c_centered.json",
        help=(
            "Where the CENTERED witness is written: the same feasible set's L1-closest "
            "point to mid-window rather than the simplex vertex."
        ),
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
    route = ThermoSource(args.route)
    if route is ThermoSource.SHIPPED:
        delta_g0_t, mask_t = gem.standard_reaction_delta_g()
        delta_g0 = delta_g0_t.double().numpy()
        mask = mask_t.numpy()
        table_sha = None
    else:
        table = resolve_reaction_delta_g(gem, THERMO_DIR, source=route)
        delta_g0 = np.asarray(table.delta_r_g0, dtype=float)
        mask = np.asarray(table.mask, dtype=bool)
        table_sha = table.sha256
    constrained = mask.copy()
    for j in exchanges:
        constrained[j] = False
    constrained[biomass] = False
    solution = model.optimize()
    assert solution.status == "optimal", f"the base FBA is {solution.status}."
    v = np.array([solution.fluxes[r] for r in rxn_ids])
    window = (CONC_LOWER_M, CONC_UPPER_M)
    full_idx = np.flatnonzero((np.abs(v) > FLUX_ACTIVE) & constrained)
    print(
        f"route {route.value}; priced {int(mask.sum())}; constrained and carrying flux "
        f"{len(full_idx)}; window {window[0]:g}..{window[1]:g} M; "
        f"loaded in {time.time() - started:.1f}s"
    )

    base_slack, slack, _ = solve(s, delta_g0, v, full_idx, window, args.epsilon_kj)
    violated = full_idx[slack > SLACK_TOLERANCE]
    print(
        f"base LP: total slack {base_slack:.1f} kJ over {len(full_idx)} constraints, "
        f"{len(violated)} violated, max {slack.max():.2f} kJ"
    )

    kept = np.array([j for j in full_idx if j not in set(violated.tolist())])
    after_l1, after_slack, after_ln_c = solve(
        s, delta_g0, v, kept, window, args.epsilon_kj
    )
    print(
        f"L1 removal ({len(violated)} reactions dropped): residual slack {after_l1:.3f} kJ "
        f"over {len(kept)} constraints"
    )
    methods: list[dict[str, object]] = [
        {
            "method": "l1_violated",
            "n_removed": int(len(violated)),
            "removed": [
                describe(model, rxn_ids, delta_g0, v, int(j)) for j in violated
            ],
            "residual_total_slack_kj": after_l1,
            "n_residual_violated": int((after_slack > SLACK_TOLERANCE).sum()),
            "meets_gate": bool(after_l1 <= args.gate_kj),
        }
    ]

    selected = methods[0]
    if after_l1 > args.gate_kj:
        # THE CONTINGENCY. The LP's L1 choice is not a minimal removal set, so a greedy
        # minimum-cardinality pass drops the single worst reaction at a time and re-solves.
        print("L1 removal missed the gate; running the greedy minimum-cardinality pass")
        current = list(full_idx)
        removed: list[int] = []
        total = base_slack
        current_slack = slack
        while total > args.gate_kj and len(removed) < GREEDY_MAX_REMOVALS:
            worst = int(current[int(np.argmax(current_slack))])
            removed.append(worst)
            current = [j for j in current if j != worst]
            total, current_slack, after_ln_c = solve(
                s, delta_g0, v, np.array(current), window, args.epsilon_kj
            )
            if len(removed) % 10 == 0:
                print(f"  removed {len(removed)}: total slack {total:.2f} kJ")
        methods.append(
            {
                "method": "greedy_max_slack",
                "n_removed": len(removed),
                "removed": [
                    describe(model, rxn_ids, delta_g0, v, int(j)) for j in removed
                ],
                "residual_total_slack_kj": float(total),
                "n_residual_violated": int((current_slack > SLACK_TOLERANCE).sum()),
                "meets_gate": bool(total <= args.gate_kj),
                "removal_cap": GREEDY_MAX_REMOVALS,
            }
        )
        selected = methods[-1]
        print(
            f"greedy: {len(removed)} reactions removed, residual slack {total:.3f} kJ, "
            f"gate met: {selected['meets_gate']}"
        )

    payload = {
        "definitions": {
            "gate": (
                f"the E2 premise holds when the total L1 slack falls below {args.gate_kj} "
                "kJ/mol once the listed reactions are exempted. Above it the infeasibility "
                "is distributed rather than concentrated and exemption is not the lever."
            ),
            "l1_violated": (
                "the reactions carrying slack at the first LP optimum. The LP minimizes "
                "the TOTAL slack, so this is where it chose to place a JOINT infeasibility "
                "and it need not be a minimal removal set."
            ),
            "greedy_max_slack": (
                "drop the single largest-slack reaction, re-solve, repeat until the gate "
                "is met. A cardinality upper bound on the true minimal set, not the "
                "minimum itself, which is an MILP."
            ),
        },
        "provenance": {
            "route": route.value,
            "thermo_table_sha256": table_sha,
            "window_m": list(window),
            "epsilon_kj": args.epsilon_kj,
            "flux": "cobra FBA at maximum biomass on the unperturbed GEM",
            "flux_active_threshold": FLUX_ACTIVE,
            "rt_kj_per_mol": RT_KJ_PER_MOL,
            "gem": osp.basename(source.model_dir),
            "n_priced": int(mask.sum()),
            "n_constrained": int(len(full_idx)),
        },
        "base_total_slack_kj": base_slack,
        "base_n_violated": int(len(violated)),
        "gate_kj": args.gate_kj,
        "selected_method": selected["method"],
        "gate_met": selected["meets_gate"],
        "exempt_reaction_ids": [row["reaction"] for row in selected["removed"]],
        "methods": methods,
    }
    out_path = osp.join(RESULTS_DIR, args.out)
    with open(out_path, "w") as handle:
        json.dump(payload, handle, indent=2)
    digest = hashlib.sha256(json.dumps(payload, indent=2).encode()).hexdigest()

    # THE CONCENTRATION WITNESS, written as its own artifact.
    #
    # WHY IT IS WORTH KEEPING. The flux layer turns the second law into a per-reaction box
    # by taking the sign of its OWN delta_r_G, which is evaluated at its OWN learned
    # concentrations. At initialization those concentrations are whatever a zero-centered
    # parameter and a random linear head produce, so the box is tightened against a
    # concentration vector that licenses nothing in particular, and a reference flux clipped
    # into it loses the direction of a third of its reactions. This vector is the LP's
    # answer to "which concentrations license this flux", so initializing the layer's log
    # concentration at it makes the directional box agree with the reference by construction
    # instead of by accident.
    #
    # It is a per-GEM quantity: it comes from this GEM's stoichiometry, this thermodynamic
    # table and this exemption list, all named below. Nothing about it is yeast-specific
    # beyond the model it was solved on.
    witness = {
        "definitions": {
            "ln_c": (
                "natural log of the concentration in molar, one entry per metabolite in "
                "the GEM's own row order. The LP's witness for the SURVIVING sign "
                "constraints, i.e. after the exempt reactions are removed."
            ),
            "validity": (
                "this vector makes the fixed flux sign-consistent only on the constraints "
                "that were still in the program, and only under the route, window and "
                "exemption list named here."
            ),
        },
        "route": route.value,
        "window_m": list(window),
        "epsilon_kj": args.epsilon_kj,
        "rt_kj_per_mol": RT_KJ_PER_MOL,
        "gem": osp.basename(source.model_dir),
        "thermo_table_sha256": table_sha,
        "exempt_file": args.out,
        "exempt_file_sha256": digest,
        "n_exempt_reactions": len(selected["removed"]),
        "n_constraints_satisfied": int(len(full_idx) - len(selected["removed"])),
        "residual_total_slack_kj": selected["residual_total_slack_kj"],
        "met_ids": list(gem.met_ids),
        "ln_c": [float(x) for x in after_ln_c],
    }
    witness_path = osp.join(RESULTS_DIR, args.witness_out)
    with open(witness_path, "w") as handle:
        json.dump(witness, handle, indent=2)

    # THE CENTERED WITNESS, the same feasible set's interior point rather than its vertex.
    surviving = np.array(
        [j for j in full_idx if rxn_ids[int(j)] not in set(payload["exempt_reaction_ids"])]
    )
    centered_ln_c, centered_slack, centered_distance = centered_witness(
        s,
        delta_g0,
        v,
        surviving,
        window,
        args.epsilon_kj,
        float(selected["residual_total_slack_kj"]),
    )
    centered = dict(witness)
    centered["ln_c"] = [float(x) for x in centered_ln_c]
    centered["objective"] = (
        "lexicographic: total slack held at the first stage's optimum, then the L1 "
        "distance of ln c from the middle of the window minimized"
    )
    centered["achieved_total_slack_kj"] = centered_slack
    centered["mean_abs_distance_from_mid_ln_c"] = centered_distance
    centered_path = osp.join(RESULTS_DIR, args.centered_out)
    with open(centered_path, "w") as handle:
        json.dump(centered, handle, indent=2)
    lo_w, hi_w = np.log(window[0]), np.log(window[1])
    print(
        f"centered witness: slack {centered_slack:.4g} kJ, mean |ln c - mid| "
        f"{centered_distance:.4f}, at a bound "
        f"{int(np.sum((np.abs(centered_ln_c - lo_w) < 1e-9) | (np.abs(centered_ln_c - hi_w) < 1e-9)))} "
        f"of {len(centered_ln_c)}"
    )
    print(f"wrote {out_path}, {witness_path} and {centered_path} in {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
