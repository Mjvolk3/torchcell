# experiments/026-metabolism-flux/scripts/probe_transport_tfa.py
# [[experiments.026-metabolism-flux.physics-gates-2026-09-14]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/026-metabolism-flux/scripts/probe_transport_tfa
r"""Does pricing the proton-motive force rescue the reactions the second law rejects?

THE OMISSION. ``FluxLayerConfig.transport_delta_g_source`` defaults to ``"off"`` and no 026
or 027 run has ever set it, so every multi-compartment reaction in every wave has been
priced as if moving a proton across a pH gradient and a membrane potential were free.
``torchcell/metabolism/transport.py`` implements Thermo-Flux Eq. (7) and
``write_transport_table`` writes it; nothing called either.

THE TEST. ``probe_tfa_lp.py`` finds that with the flux fixed at a real FBA solution, 47
priced reactions cannot be made downhill by any concentration in the physiological window
under the ``union_external`` route, and ATP synthase (``r_0226``) is among them at a slack
of about 10 kJ/mol, which is the order of one proton across the mitochondrial gradient.
This script adds the computed transport term to the same standard energies and re-solves
that LP.

**Read:** does the violated count fall below the no-transport count, and does ``r_0226``
leave the list? If neither moves, the transport term is a correctness fix for the writeup
and not a Genesis lever, and E3 does not get a GPU.

TWO TABLES, BECAUSE THE pH COVERAGE IS THE WHOLE CAVEAT.

``three_compartment``
    pH for ``c``, ``m`` and ``e`` only, which is what ``FluxLayerConfig``'s
    ``CompartmentParameters`` actually tabulates. Every other compartment falls back to the
    reference pH, so a reaction crossing one gets no pH term at all. This is the honest
    statement of what the layer can price today.

``pipeline``
    ``transport.pipeline_conditions()``, which assumes textbook organellar pH for all 14
    compartments. Its extra rows are LABELED ASSUMPTIONS, not measurements, and the count of
    reactions that depend on them is reported beside the result.

Run from the repo root::

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=$PWD ~/miniconda3/envs/torchcell/bin/python \
        experiments/026-metabolism-flux/scripts/probe_transport_tfa.py --route union_external

Writes ``experiments/026-metabolism-flux/results/probe_transport_tfa.json`` and the two
transport tables into ``$DATA_ROOT/data/torchcell/transport/``.
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
from torchcell.metabolism.transport import (  # noqa: E402
    CompartmentElectrochemistry,
    TransportTable,
    pipeline_conditions,
    write_transport_table,
)
from torchcell.metabolism.yeast_GEM import YeastGEM  # noqa: E402

DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "026-metabolism-flux", "results")
THERMO_DIR = osp.join(DATA_ROOT, "data/torchcell/thermo_equilibrator")
TRANSPORT_DIR = osp.join(DATA_ROOT, "data/torchcell/transport")

RT_KJ_PER_MOL = 8.314462618e-3 * 303.15
FLUX_ACTIVE = 1.0e-6
SLACK_TOLERANCE = 1.0e-6
#: The reaction the review names as the most likely mechanistic beneficiary.
WATCHED_REACTION = "r_0226"


def three_compartment_conditions() -> CompartmentElectrochemistry:
    """Exactly what ``FluxLayerConfig``'s ``CompartmentParameters`` carries.

    pH for the cytosol, the mitochondrion and the extracellular space, Thermo-Flux's
    iMM904 Table 1; every other compartment is untabulated and therefore takes the
    reference pH, which zeroes its pH term rather than guessing one. The potentials are
    the same table's two magnitudes with the signs transport.py derives.
    """
    p_h = {"c": 7.0, "m": 7.4, "e": 5.0}
    return CompartmentElectrochemistry(
        p_h=p_h,
        ionic_strength_m={k: 0.25 for k in p_h},
        potential_mv={"e": 60.0, "m": -160.0},
        reference_compartment="c",
        temperature_k=303.15,
        source=(
            "Smith et al. 2026, Thermo-Flux, Table 1 (iMM904 parameter set), as carried "
            "by torchcell.metabolism.flux_layer.CompartmentParameters. Three compartments "
            "only; every other compartment is untabulated and takes no pH term."
        ),
    )


def slack_lp(
    s: np.ndarray,
    delta_g0: np.ndarray,
    v: np.ndarray,
    idx: np.ndarray,
    window_m: tuple[float, float],
    epsilon_kj: float,
) -> tuple[float, np.ndarray]:
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
    return float(program.fun), program.x[n_met:]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--route",
        default=ThermoSource.UNION_EXTERNAL.value,
        choices=[s.value for s in ThermoSource],
    )
    parser.add_argument("--epsilon-kj", type=float, default=1.0e-3)
    parser.add_argument("--out", default="probe_transport_tfa.json")
    args = parser.parse_args()

    started = time.time()
    os.makedirs(TRANSPORT_DIR, exist_ok=True)
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
        base = delta_g0_t.double().numpy()
        mask = mask_t.numpy()
    else:
        table = resolve_reaction_delta_g(gem, THERMO_DIR, source=route)
        base = np.asarray(table.delta_r_g0, dtype=float)
        mask = np.asarray(table.mask, dtype=bool)
    constrained = mask.copy()
    for j in exchanges:
        constrained[j] = False
    constrained[biomass] = False
    solution = model.optimize()
    assert solution.status == "optimal", f"the base FBA is {solution.status}."
    v = np.array([solution.fluxes[r] for r in rxn_ids])
    window = (CONC_LOWER_M, CONC_UPPER_M)
    idx = np.flatnonzero((np.abs(v) > FLUX_ACTIVE) & constrained)
    print(
        f"route {route.value}; constrained and carrying flux {len(idx)}; "
        f"loaded in {time.time() - started:.1f}s"
    )

    variants: list[dict[str, object]] = []
    tables = {
        "off": None,
        "three_compartment": three_compartment_conditions(),
        "pipeline": pipeline_conditions(),
    }
    for name, conditions in tables.items():
        transport = np.zeros(len(rxn_ids))
        coverage: dict[str, object] = {}
        path = None
        if conditions is not None:
            path = write_transport_table(
                model,
                TRANSPORT_DIR,
                conditions,
                filename=f"transport_terms_{name}.json",
            )
            written = TransportTable.read(path)
            values, has_term = written.as_tensor(rxn_ids)
            transport = values.numpy().astype(float)
            tabulated = set(conditions.p_h)
            crossing = [t for t in written.terms if t.crosses_membrane]
            untabulated = [
                t
                for t in crossing
                if any(
                    c not in tabulated for pair in t.compartment_pairs for c in pair
                )
            ]
            coverage = {
                "table_path": path,
                "n_reactions_with_a_term": int(has_term.sum()),
                "n_crossing_a_membrane": len(crossing),
                "n_nonzero_terms": int((np.abs(transport) > 1e-9).sum()),
                "n_crossing_an_untabulated_compartment": len(untabulated),
                "tabulated_compartments": sorted(tabulated),
                "assumption": (
                    "a reaction crossing a compartment with no tabulated pH takes the "
                    "reference compartment's pH, so its pH term is exactly zero. That is "
                    "an ASSUMPTION, not a measurement, and the count above is how many "
                    "reactions rest on it."
                ),
                "conditions_source": conditions.source,
            }
        total, slack = slack_lp(
            s, base + transport, v, idx, window, args.epsilon_kj
        )
        violated = idx[slack > SLACK_TOLERANCE]
        violated_ids = [rxn_ids[int(j)] for j in violated]
        order = np.argsort(-slack)
        row = {
            "transport_source": name,
            "n_constrained": int(len(idx)),
            "n_violated": int(len(violated)),
            "total_slack_kj": total,
            "max_violation_kj": float(slack.max()),
            "watched_reaction": WATCHED_REACTION,
            "watched_reaction_violated": WATCHED_REACTION in violated_ids,
            "watched_reaction_slack_kj": (
                float(slack[list(idx).index(rxn_ids.index(WATCHED_REACTION))])
                if WATCHED_REACTION in [rxn_ids[int(j)] for j in idx]
                else None
            ),
            "watched_reaction_transport_kj": float(
                transport[rxn_ids.index(WATCHED_REACTION)]
            ),
            "violated_reaction_ids": violated_ids,
            "worst": [
                {
                    "reaction": rxn_ids[int(idx[k])],
                    "slack_kj": float(slack[k]),
                    "delta_r_g0_kj": float(base[int(idx[k])]),
                    "transport_kj": float(transport[int(idx[k])]),
                    "v": float(v[int(idx[k])]),
                }
                for k in order[:15]
                if slack[k] > SLACK_TOLERANCE
            ],
            "coverage": coverage,
        }
        variants.append(row)
        print(
            f"{name:18s} violated {row['n_violated']:3d}  L1 {total:10.1f} kJ  "
            f"max {row['max_violation_kj']:8.2f}  {WATCHED_REACTION} violated: "
            f"{row['watched_reaction_violated']}"
        )

    baseline = variants[0]
    payload = {
        "definitions": {
            "read": (
                "the gate is whether the violated count falls below the transport-off "
                "count and whether r_0226 (ATP synthase) leaves the violated list."
            ),
            "watched_reaction_slack_kj": (
                "the slack the LP puts on r_0226 at its optimum, which is zero when the "
                "reaction is satisfiable."
            ),
        },
        "provenance": {
            "route": route.value,
            "window_m": list(window),
            "epsilon_kj": args.epsilon_kj,
            "flux": "cobra FBA at maximum biomass on the unperturbed GEM",
            "rt_kj_per_mol": RT_KJ_PER_MOL,
            "gem": osp.basename(source.model_dir),
        },
        "baseline_n_violated": baseline["n_violated"],
        "gate_met": any(
            row["n_violated"] < baseline["n_violated"] for row in variants[1:]
        ),
        "watched_reaction_freed_by": [
            row["transport_source"]
            for row in variants[1:]
            if not row["watched_reaction_violated"]
        ],
        "variants": variants,
    }
    out_path = osp.join(RESULTS_DIR, args.out)
    with open(out_path, "w") as handle:
        json.dump(payload, handle, indent=2)
    print(f"wrote {out_path} in {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
