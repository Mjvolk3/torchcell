# experiments/026-metabolism-flux/scripts/pathway_thermo.py
# [[experiments.026-metabolism-flux.scripts.pathway_thermo]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/026-metabolism-flux/scripts/pathway_thermo.py

r"""Formation energies for a heterologous pathway's metabolites, and where they run out.

The betaxanthin cassette is the test case for the claim that eQuilibrator generalizes: its
intermediates are plant metabolites that yeast-GEM has never carried, so if a structure is
all that is needed, they should resolve like any other compound.

They partly do, and the boundary is sharp and worth stating precisely.

* An intermediate that already sits in the eQuilibrator cache resolves by structure and
  gets a formation energy with uncertainty, no identifier required.
* An intermediate that does NOT sit in the cache cannot be added. Creating a compound runs
  a group decomposition that needs pKa estimates, and eQuilibrator obtains those from
  ChemAxon's ``cxcalc``, which is commercial. Without that license
  ``equilibrator_assets`` loads in read-only mode and refuses to create compounds.

So the generalization claim holds for structures the cache already knows and fails for
genuinely new ones, and the blocker is a license rather than a method. That distinction
decides what to do next: a novel-compound route has to come from a decomposition that
estimates pKa without ChemAxon, not from more plumbing here.

Structures are retrieved from PubChem by name and checked against the molecular formulas
the pathway module declares, so a wrong-compound hit shows up as a formula mismatch
instead of a plausible number.
"""

import argparse
import json
import os
import os.path as osp
import urllib.parse
import urllib.request
from datetime import UTC, datetime

from equilibrator_api import ComponentContribution, Q_

# Molecular formulas as declared in torchcell.metabolism.betaxanthin, used to verify that
# the PubChem name lookup returned the intended compound.
PATHWAY_COMPOUNDS: list[tuple[str, str, str]] = [
    ("L-DOPA", "C9H11NO4", "CYP76AD1 hydroxylation product"),
    ("dopaquinone", "C9H9NO4", "CYP76AD1 oxidase side activity"),
    ("leucodopachrome", "C9H9NO4", "cyclo-DOPA, spontaneous cyclization"),
    ("betanidin", "C18H16N2O8", "violet pigment, competing branch"),
    ("betalamic acid", "C9H9NO5", "DOD product, the reactive aldehyde"),
]

PUBCHEM = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}"
    "/property/SMILES,MolecularFormula/JSON"
)


def pubchem_structure(name: str) -> tuple[str | None, str | None]:
    """SMILES and molecular formula for a compound name, or (None, None)."""
    url = PUBCHEM.format(name=urllib.parse.quote(name))
    try:
        with urllib.request.urlopen(url, timeout=45) as response:
            payload = json.load(response)
    except Exception:
        return None, None
    record = payload["PropertyTable"]["Properties"][0]
    smiles = record.get("SMILES") or record.get("CanonicalSMILES")
    return smiles, record.get("MolecularFormula")


def main() -> None:
    """Resolve each pathway intermediate and report its energy or its blocker."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--p-h", type=float, default=7.2)
    args = parser.parse_args()

    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")

    cc = ComponentContribution()
    cc.p_h = Q_(args.p_h)
    cc.ionic_strength = Q_("0.25M")
    cc.temperature = Q_("303.15K")
    cc.p_mg = Q_(3.0)

    rows = []
    for name, expected_formula, role in PATHWAY_COMPOUNDS:
        smiles, formula = pubchem_structure(name)
        row = {
            "name": name,
            "role": role,
            "expected_formula": expected_formula,
            "pubchem_formula": formula,
            "formula_matches": formula == expected_formula,
            "smiles": smiles,
            "in_equilibrator_cache": False,
            "dgf_prime_kj_per_mol": None,
            "sigma_kj_per_mol": None,
            "blocker": None,
        }
        if smiles is None:
            row["blocker"] = "pubchem_name_lookup_failed"
            rows.append(row)
            continue
        if formula != expected_formula:
            row["blocker"] = "formula_mismatch_wrong_compound"
            rows.append(row)
            continue

        molecule = Chem.MolFromSmiles(smiles)
        compound = (
            cc.get_compound_by_inchi(Chem.MolToInchi(molecule))
            if molecule is not None
            else None
        )
        if compound is None:
            row["blocker"] = "absent_from_cache_needs_cxcalc_chemaxon_license"
            rows.append(row)
            continue

        row["in_equilibrator_cache"] = True
        mu, sigma_fin, _ = cc.standard_dg_formation(compound)
        if mu is None:
            row["blocker"] = "in_cache_but_no_formation_energy"
            rows.append(row)
            continue
        correction = compound.transform(
            Q_(args.p_h), Q_("0.25M"), Q_("303.15K"), Q_(3.0)
        )
        row["dgf_prime_kj_per_mol"] = float(mu) + float(correction.m_as("kJ/mol"))
        import numpy as np

        row["sigma_kj_per_mol"] = float(np.linalg.norm(sigma_fin))
        rows.append(row)

    resolved = sum(1 for r in rows if r["dgf_prime_kj_per_mol"] is not None)
    summary = {
        "pathway": "betaxanthin (DeLoache et al. 2015)",
        "n_compounds": len(rows),
        "n_resolved": resolved,
        "conditions": {
            "p_h": args.p_h,
            "ionic_strength_m": 0.25,
            "temperature_k": 303.15,
            "p_mg": 3.0,
        },
        "compounds": rows,
        "built_at": datetime.now(UTC).isoformat(),
    }

    results = osp.join(
        os.environ["EXPERIMENT_ROOT"], "026-metabolism-flux", "results"
    )
    os.makedirs(results, exist_ok=True)
    with open(osp.join(results, "pathway_thermo_betaxanthin.json"), "w") as handle:
        json.dump(summary, handle, indent=2)

    print(f"{'compound':18s} {'formula ok':>10s} {'dfG prime':>12s} {'sigma':>8s}  blocker")
    for row in rows:
        value = (
            f"{row['dgf_prime_kj_per_mol']:12.2f}"
            if row["dgf_prime_kj_per_mol"] is not None
            else f"{'--':>12s}"
        )
        sigma = (
            f"{row['sigma_kj_per_mol']:8.2f}"
            if row["sigma_kj_per_mol"] is not None
            else f"{'--':>8s}"
        )
        print(
            f"{row['name']:18s} {str(row['formula_matches']):>10s} {value} {sigma}  "
            f"{row['blocker'] or ''}"
        )
    print(f"\n{resolved} of {len(rows)} resolved")


if __name__ == "__main__":
    main()
