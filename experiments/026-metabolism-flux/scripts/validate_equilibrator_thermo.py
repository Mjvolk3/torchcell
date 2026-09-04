# experiments/026-metabolism-flux/scripts/validate_equilibrator_thermo.py
# [[experiments.026-metabolism-flux.scripts.validate_equilibrator_thermo]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/026-metabolism-flux/scripts/validate_equilibrator_thermo.py

r"""Check the formation-energy table reproduces eQuilibrator's own reaction call.

The table stores per-metabolite :math:`\Delta_f G'^\circ` because that is what the flux
layer consumes, and reaction energies come from :math:`\sum_i S_{ij}\Delta_f G'^\circ_i`.
That identity is an assumption until it is tested: an error in the Legendre transform, in
the compartment pH assignment, or in the sign of a stoichiometric coefficient would all
produce a table that looks entirely reasonable and is wrong.

So the same reactions are priced both ways, by summing our table and by handing the
resolved compounds to ``standard_dg_prime``, and the two are compared. Only
single-compartment reactions are used, because a reaction spanning two compartments has no
single pH and the API call would not be answering the same question. Only mass-balanced
reactions are used, because an unbalanced one has an arbitrary offset that cancels in
neither calculation.

Run under the eQuilibrator environment:
    /scratch/projects/torchcell-scratch/envs/equilibrator/bin/python \
        experiments/026-metabolism-flux/scripts/validate_equilibrator_thermo.py
"""

import argparse
import json
import os
import os.path as osp
import random
import sys
from datetime import UTC, datetime

import cobra
import numpy as np
import pandas as pd
from equilibrator_api import ComponentContribution, Q_, Reaction

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from build_equilibrator_thermo import (  # noqa: E402
    COMPARTMENT_PH,
    GEM_DIR,
    IONIC_STRENGTH_M,
    P_MG,
    TEMPERATURE_K,
    read_smiles_db,
    resolve_compound,
)

THERMO = osp.join(
    os.environ["DATA_ROOT"], "data", "torchcell", "thermo_equilibrator"
)
RESULTS = osp.join(
    os.environ["EXPERIMENT_ROOT"], "026-metabolism-flux", "results"
)

# Textbook reactions priced directly through the API, as an absolute sanity check that
# eQuilibrator itself is configured correctly before anything is compared against it.
REFERENCE_REACTIONS = {
    "ATP hydrolysis": "kegg:C00002 + kegg:C00001 = kegg:C00008 + kegg:C00009",
    "phosphoglucose isomerase": "kegg:C00092 = kegg:C00085",
    "fumarase": "kegg:C00122 + kegg:C00001 = kegg:C00149",
    "triosephosphate isomerase": "kegg:C00111 = kegg:C00118",
}


def main() -> None:
    """Price reactions both ways and write the comparison."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-reactions", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    metabolites = pd.read_parquet(osp.join(THERMO, "metabolite_dgf_prime.parquet"))
    reactions = pd.read_parquet(osp.join(THERMO, "reaction_drg_prime.parquet"))
    model = cobra.io.read_sbml_model(osp.join(GEM_DIR, "model", "yeast-GEM.xml"))
    smiles_db = read_smiles_db(osp.join(GEM_DIR, "data", "databases", "smilesDB.tsv"))

    cc = ComponentContribution()
    cc.ionic_strength = Q_(f"{IONIC_STRENGTH_M}M")
    cc.temperature = Q_(f"{TEMPERATURE_K}K")
    cc.p_mg = Q_(P_MG)

    reference = {}
    cc.p_h = Q_(COMPARTMENT_PH["c"])
    for name, formula in REFERENCE_REACTIONS.items():
        measurement = cc.standard_dg_prime(cc.parse_reaction_formula(formula))
        reference[name] = {
            "value_kj_per_mol": float(measurement.value.m_as("kJ/mol")),
            "error_kj_per_mol": float(measurement.error.m_as("kJ/mol")),
        }

    covered = set(reactions.loc[reactions["all_participants_known"], "reaction_id"])
    ours = reactions.set_index("reaction_id")["drg_prime_kj_per_mol"]

    candidates = []
    for reaction in model.reactions:
        if reaction.id not in covered:
            continue
        compartments = {m.compartment for m in reaction.metabolites}
        if len(compartments) != 1:
            continue
        if reaction.check_mass_balance():
            continue
        candidates.append(reaction)
    random.seed(args.seed)
    random.shuffle(candidates)

    cache: dict[str, object] = {}
    rows = []
    for reaction in candidates:
        if len(rows) >= args.n_reactions:
            break
        compartment = next(iter({m.compartment for m in reaction.metabolites}))
        cc.p_h = Q_(COMPARTMENT_PH[compartment])
        stoichiometry: dict[object, float] = {}
        resolved = True
        for metabolite, coefficient in reaction.metabolites.items():
            key = (metabolite.name or metabolite.id).lower()
            if key not in cache:
                cache[key] = resolve_compound(cc, metabolite, smiles_db)[0]
            compound = cache[key]
            if compound is None:
                resolved = False
                break
            stoichiometry[compound] = stoichiometry.get(compound, 0) + coefficient
        if not resolved or not stoichiometry:
            continue
        api = cc.standard_dg_prime(Reaction(stoichiometry)).value.m_as("kJ/mol")
        rows.append(
            {
                "reaction_id": reaction.id,
                "compartment": compartment,
                "ours": float(ours[reaction.id]),
                "api": float(api),
            }
        )

    differences = np.array([r["ours"] - r["api"] for r in rows])
    summary = {
        "n_candidates": len(candidates),
        "n_compared": len(rows),
        "max_abs_diff": float(np.abs(differences).max()) if len(rows) else float("nan"),
        "median_abs_diff": (
            float(np.median(np.abs(differences))) if len(rows) else float("nan")
        ),
        "conditions": {
            "temperature_k": TEMPERATURE_K,
            "ionic_strength_m": IONIC_STRENGTH_M,
            "p_mg": P_MG,
            "p_h": "per compartment",
        },
        "reference_reactions": reference,
        "reactions": rows,
        "checked_at": datetime.now(UTC).isoformat(),
    }
    os.makedirs(RESULTS, exist_ok=True)
    with open(osp.join(RESULTS, "equilibrator_api_validation.json"), "w") as handle:
        json.dump(summary, handle, indent=2)

    print(f"{'reaction':>26s} {'value':>10s} {'error':>7s}")
    for name, values in reference.items():
        print(
            f"{name:>26s} {values['value_kj_per_mol']:10.2f} "
            f"{values['error_kj_per_mol']:7.2f}"
        )
    print(
        f"\ncompared {len(rows)} single-compartment balanced reactions; "
        f"max |diff| {summary['max_abs_diff']:.3g} kJ/mol"
    )


if __name__ == "__main__":
    main()
