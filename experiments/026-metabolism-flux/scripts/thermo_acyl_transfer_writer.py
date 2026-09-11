# experiments/026-metabolism-flux/scripts/thermo_acyl_transfer_writer.py
# [[experiments.026-metabolism-flux.scripts.thermo_acyl_transfer_writer]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/026-metabolism-flux/scripts/thermo_acyl_transfer_writer.py

r"""Compose a product's structure from the REACTION that makes it.

**A structure writer does not have to read the name.** The lipid writer composes a structure
from a name because yeast-GEM writes acyl composition into lipid names. A second and
stronger source of the same information is the reaction: in an acyl transfer

    acyl-CoA + R-OH  ->  CoA + R-O-acyl

the product is the substrate alcohol with one hydrogen replaced by the acyl group, and BOTH
the alcohol and the acyl donor are already in the model with structures. So the product's
structure is determined by the stoichiometry, and nothing has to be guessed from a name.

**This corrects a structure that was wrong.** ``external_pka_pilot.json`` carried
``methionyl acetate`` as ``CC(=O)OC(=O)C(CCSC)N``, the mixed anhydride of METHIONINE, whose
formula is ``C7H13NO3S``. yeast-GEM declares ``C6H12O2S`` and makes the species in
``r_4690, alcohol acetyltransferase (methionol)`` from METHIONOL, ``C4H10OS``. "Methionyl"
was read as the methionine acyl group when the model means the methionol alkyl group, and
the pilot had no formula acceptance test to catch it. The lipid writer has one; that test is
applied here, and a composed structure whose RDKit formula does not reproduce the GEM's is
rejected rather than priced.

**Which hydroxyl is acylated is decided by chemistry, not by position.** Tyrosol carries an
aliphatic hydroxyl and a phenol, and both give the same molecular formula, so the formula
test cannot separate them. The enzyme is an ALCOHOL acetyltransferase, so the SMARTS matches
only a hydroxyl on an sp3 carbon. A substrate with more than one such hydroxyl is rejected
as ambiguous rather than resolved by picking the first.

One subcommand, ``write``, run under the eQuilibrator environment, which carries both cobra
and RDKit. It composes and TESTS structures and prices nothing: pricing is
``external_pka_apply_selection.py``'s job, which reads this file as one more source of
targets so the selected pKa method and the pinned protonation convention are applied in
exactly one place.

Writes ``results/thermo_acyl_transfer.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
from datetime import datetime, timezone

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
GEM_DIR = osp.join(DATA_ROOT, "data", "torchcell", "yeast-GEM", "yeast-GEM-9.0.2")
RESULTS = osp.join(EXPERIMENT_ROOT, "026-metabolism-flux", "results")
PIPELINE_JSON = osp.join(RESULTS, "thermo_pipeline_yeast9.json")
OUT_PATH = osp.join(RESULTS, "thermo_acyl_transfer.json")

P_H = 7.2
IONIC_STRENGTH_M = 0.25
TEMPERATURE_K = 303.15
P_MG = 3.0

#: A hydroxyl on an sp3 carbon: the ALCOHOL an alcohol acyltransferase acts on, and not a
#: phenol, a carboxylic acid or an enol.
ALIPHATIC_HYDROXYL = "[CX4][OX2H]"

#: The acyl donors this writer understands, as (metabolite id, acyl SMILES fragment). The
#: fragment is what replaces the hydroxyl hydrogen. Acetyl-CoA is the only one yeast-GEM's
#: unpriced products need; the mapping is data so another donor is a line, not a branch.
ACYL_DONORS: dict[str, dict[str, str]] = {
    "s_0373": {"name": "acetyl-CoA", "acyl_smiles": "C(C)=O", "released": "s_0529"},
}


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()  # noqa: UP017


class ComposedProduct(BaseModel):
    """One product composed from its own reaction, and whether the formula test passed."""

    met_id: str
    met_ids: list[str] = Field(default_factory=list)
    name: str
    gem_formula: str
    gem_charge: int
    reaction_id: str
    reaction_name: str
    reaction_string: str
    donor_met_id: str
    donor_name: str
    acceptor_met_id: str
    acceptor_name: str
    acceptor_smiles: str | None = None
    product_smiles: str | None = None
    rdkit_formula: str | None = None
    neutral_target_formula: str | None = None
    formula_match: bool = False
    outcome: str = ""
    detail: str = ""


class AcylTransferReport(BaseModel):
    """What the reaction-driven writer composed, and what passed the formula test."""

    built_at: str
    conditions: dict[str, float]
    acceptance_test: str = (
        "the RDKit formula of the composed structure must equal the GEM formula moved to "
        "the neutral species, H_neutral = H_gem - charge_gem with every other element "
        "unchanged; anything else is a different molecule and is rejected"
    )
    hydroxyl_smarts: str = ALIPHATIC_HYDROXYL
    n_candidate_reactions: int
    n_composed: int
    n_formula_pass: int
    n_priced: int
    n_metabolite_rows_priced: int
    outcomes: dict[str, int] = Field(default_factory=dict)
    records: list[ComposedProduct] = Field(default_factory=list)


def _neutral_target_formula(formula: str, charge: int) -> str:
    """The GEM formula moved to the neutral species, the lipid writer's own convention."""
    import re

    if charge == 0:
        return formula
    tokens = re.findall(r"([A-Z][a-z]?)(\d*)", formula)
    counts = {
        element: int(count) if count else 1
        for element, count in tokens
        if element
    }
    counts["H"] = counts.get("H", 0) - charge
    return "".join(
        f"{element}{counts[element] if counts[element] != 1 else ''}"
        for element in sorted(counts)
        if counts[element] > 0
    )


def cmd_write(args: argparse.Namespace) -> None:
    """Compose every unpriced acyl-transfer product and run the formula acceptance test."""
    import cobra
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdMolDescriptors
    from rdkit.Chem.rdChemReactions import ReactionFromSmarts

    RDLogger.DisableLog("rdApp.*")

    with open(PIPELINE_JSON) as handle:
        pipeline = json.load(handle)
    unpriced = {
        row["met_id"] for row in pipeline["metabolites"] if row["dgf_prime_kj_per_mol"] is None
    }
    model = cobra.io.read_sbml_model(osp.join(GEM_DIR, "model", "yeast-GEM.xml"))

    from equilibrator_api import ComponentContribution

    cc = ComponentContribution()

    def _smiles_for(met) -> str | None:
        annotation = met.annotation or {}
        for key in ("bigg.metabolite", "chebi", "kegg.compound", "metanetx.chemical"):
            values = annotation.get(key)
            if not values:
                continue
            for value in values if isinstance(values, list) else [values]:
                try:
                    compound = cc.ccache.get_compound(f"{key}:{value}")
                except Exception:
                    continue
                if compound is not None and compound.smiles:
                    return str(compound.smiles)
        return None

    records: list[ComposedProduct] = []
    n_candidates = 0
    for reaction in model.reactions:
        donors = [
            m for m in reaction.metabolites
            if m.id in ACYL_DONORS and reaction.metabolites[m] < 0
        ]
        if not donors:
            continue
        products = [
            m
            for m in reaction.metabolites
            if reaction.metabolites[m] > 0 and m.id in unpriced
        ]
        if len(products) != 1:
            continue
        n_candidates += 1
        donor = donors[0]
        spec = ACYL_DONORS[donor.id]
        acceptors = [
            m
            for m in reaction.metabolites
            if reaction.metabolites[m] < 0 and m.id != donor.id
        ]
        product = products[0]
        record = ComposedProduct(
            met_id=product.id,
            met_ids=[product.id],
            name=product.name,
            gem_formula=product.formula or "",
            gem_charge=int(product.charge or 0),
            reaction_id=reaction.id,
            reaction_name=reaction.name,
            reaction_string=reaction.reaction[:160],
            donor_met_id=donor.id,
            donor_name=spec["name"],
            acceptor_met_id=acceptors[0].id if acceptors else "",
            acceptor_name=acceptors[0].name if acceptors else "",
        )
        if len(acceptors) != 1:
            record.outcome = "ambiguous_acceptor"
            record.detail = f"{len(acceptors)} substrates other than the acyl donor"
            records.append(record)
            continue
        acceptor_smiles = _smiles_for(acceptors[0])
        record.acceptor_smiles = acceptor_smiles
        if acceptor_smiles is None:
            record.outcome = "no_acceptor_structure"
            records.append(record)
            continue
        mol = Chem.MolFromSmiles(acceptor_smiles)
        pattern = Chem.MolFromSmarts(ALIPHATIC_HYDROXYL)
        matches = mol.GetSubstructMatches(pattern) if mol is not None else ()
        if len(matches) != 1:
            record.outcome = "ambiguous_hydroxyl"
            record.detail = (
                f"{len(matches)} aliphatic hydroxyls; which one the enzyme acylates is not "
                "determined by the formula, so nothing is composed"
            )
            records.append(record)
            continue
        rxn = ReactionFromSmarts(
            f"[CX4:1][OX2H:2]>>[C:1][O:2]{spec['acyl_smiles']}"
        )
        outcomes = rxn.RunReactants((mol,))
        if not outcomes:
            record.outcome = "acylation_failed"
            records.append(record)
            continue
        composed = outcomes[0][0]
        try:
            Chem.SanitizeMol(composed)
        except Exception as exc:
            record.outcome = "composed_structure_does_not_sanitize"
            record.detail = f"{type(exc).__name__}"
            records.append(record)
            continue
        record.product_smiles = Chem.MolToSmiles(composed)
        record.rdkit_formula = rdMolDescriptors.CalcMolFormula(composed).rstrip("+-")
        record.neutral_target_formula = _neutral_target_formula(
            record.gem_formula, record.gem_charge
        )
        record.formula_match = record.rdkit_formula == record.neutral_target_formula
        record.outcome = "composed" if record.formula_match else "formula_mismatch"
        if not record.formula_match:
            record.detail = (
                f"composed {record.rdkit_formula}, the GEM declares "
                f"{record.neutral_target_formula}"
            )
        records.append(record)
        print(
            f"  {product.id} {product.name[:34]:34s} {record.outcome}"
            f" {record.rdkit_formula} vs {record.neutral_target_formula}",
            flush=True,
        )

    # A transported or exchanged copy of an accepted product is the same chemical, so it
    # takes the same structure. Pulling those in here is what makes the stage price the
    # extracellular copy as well as the cytosolic one.
    by_name: dict[str, ComposedProduct] = {
        r.name: r for r in records if r.formula_match
    }
    for met in model.metabolites:
        if met.id in unpriced and met.name in by_name:
            record = by_name[met.name]
            if met.id not in record.met_ids:
                record.met_ids.append(met.id)

    outcomes_count: dict[str, int] = {}
    for record in records:
        outcomes_count[record.outcome] = outcomes_count.get(record.outcome, 0) + 1
    report = AcylTransferReport(
        built_at=_timestamp(),
        conditions={
            "p_h": P_H,
            "ionic_strength_m": IONIC_STRENGTH_M,
            "temperature_k": TEMPERATURE_K,
            "p_mg": P_MG,
        },
        n_candidate_reactions=n_candidates,
        n_composed=sum(1 for r in records if r.product_smiles),
        n_formula_pass=sum(1 for r in records if r.formula_match),
        n_priced=0,
        n_metabolite_rows_priced=0,
        outcomes=outcomes_count,
        records=records,
    )
    with open(OUT_PATH, "w") as handle:
        handle.write(report.model_dump_json(indent=1))
    print(f"wrote {OUT_PATH}: {report.n_formula_pass} structures passed the formula test")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_write = sub.add_parser("write", help="compose the products and check their formulas")
    p_write.set_defaults(func=cmd_write)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
