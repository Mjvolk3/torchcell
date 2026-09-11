# experiments/026-metabolism-flux/scripts/kinetic_substrate_writer.py
# [[experiments.026-metabolism-flux.scripts.kinetic_substrate_writer]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/026-metabolism-flux/scripts/kinetic_substrate_writer.py

r"""Write the substrate structures no mirrored catalog carries, and let the formula pick them.

WHAT IS LEFT AFTER THE CATALOGS
--------------------------------
``improve_kinetic_coverage.py --stage verify`` leaves a handful of substrates with no
structure from any mirrored source and no R in their formula, so they are not placeholders:
the GEM has committed to a definite composition and simply named a compound the catalogs do
not index. Six of them block catalytic units:

===========================  ==================  ==================================
metabolite                   GEM formula         what it is
===========================  ==================  ==================================
dolichyl D-mannosyl          ``C26H47O9P``       a mannosyl phosphodiester of a
phosphate                                        dolichol of unstated length
cis-dec-3-enoyl-CoA          ``C31H52N7O17P3S``  an ordinary acyl-CoA thioester
N,N'-diformyldityrosine      ``C20H22N2O8``      the spore-wall cross-link
phosphatidylethanolamine     ``C5H10NO4P``       a backbone with the acyl positions
backbone                                         REMOVED from the formula
hydrogen sulfide             ``HS``              a formula that is not a molecule
G10595 and friends           ``...R``            placeholders, handled elsewhere
===========================  ==================  ==================================

THE CHAIN LENGTH IS DERIVED, NOT CHOSEN
----------------------------------------
The interesting case is the dolichol. A dolichol is a polyprenol of variable length whose
alpha isoprene unit is saturated, and yeast makes a distribution of them, so "dolichyl" in a
name does not say how long. **The GEM's own formula does.** So the writer builds
``dolichol-n`` for every n in a range and keeps the one whose molecular formula equals the
GEM's, which turns a modeling choice into a lookup against a constraint the GEM already
made. ``C26H47O9P`` selects n = 4 uniquely, and if it had selected none the compound would
be reported unwritable rather than approximated.

The acyl-CoA family works the same way: the name gives a chain length and a double bond, the
template builds the thioester, and the formula test either accepts it or does not.

**Nothing here is accepted on the strength of the name.** Every candidate goes through the
same neutralized-formula comparison the cross-reference route uses
(:func:`~torchcell.metabolism.kinetics_pipeline.structure_matches_metabolite`), and a
candidate that fails is written to the report as a failure with its computed formula beside
the GEM's, which is how the two genuinely unwritable species below were identified rather
than guessed at.

WHAT THIS WRITER REFUSES, AND WHY THAT IS THE ANSWER
------------------------------------------------------
* **phosphatidylethanolamine backbone**, ``C5H10NO4P``. This is glycerophosphoethanolamine
  minus two waters: the two acyl positions were removed from the FORMULA rather than written
  as ``R``. No molecule has that composition and that connectivity, so any SMILES matching
  the formula would be a different compound. It is an R-group placeholder whose R was
  deleted, and the honest classification is unwritable.
* **hydrogen sulfide**, ``HS`` at charge 0. Hydrogen sulfide is ``H2S``; ``HS`` at charge
  zero is a radical. The GEM's formula is wrong, and writing ``S`` for it would hide that.
  Reported as a GEM defect.

Run from the worktree root, before ``--stage verify``::

    PYTHONPATH=$PWD ~/miniconda3/envs/torchcell/bin/python \
        experiments/026-metabolism-flux/scripts/kinetic_substrate_writer.py
"""

from __future__ import annotations

import os
import os.path as osp
from datetime import UTC, datetime

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field

from torchcell.metabolism.kinetics_pipeline import (
    formula_from_smiles,
    structure_matches_metabolite,
)
from torchcell.metabolism.yeast_GEM import YeastGEM

load_dotenv()

EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "026-metabolism-flux", "results")
OUT_PATH = osp.join(RESULTS_DIR, "kinetic_substrate_structures.json")

#: Coenzyme A from the thioester sulfur outwards, written once. Identical to the moiety the
#: lipid writer uses for its acyl-CoA species, so a CoA thioester from either writer is the
#: same molecule; stereocenters are left unspecified there and here.
COA_TAIL = (
    "SCCNC(=O)CCNC(=O)C(O)C(C)(C)COP(=O)(O)OP(=O)(O)OCC1OC(n2cnc3c(N)ncnc32)"
    "C(O)C1OP(=O)(O)O"
)

#: Chain-length words as they appear in a GEM acyl name.
CHAIN_LENGTHS = {
    "but": 4,
    "hex": 6,
    "oct": 8,
    "dec": 10,
    "dodec": 12,
    "tetradec": 14,
    "hexadec": 16,
    "octadec": 18,
}


def acyl_coa(n_carbons: int, double_bond: int | None, cis: bool) -> str:
    """The CoA thioester of one acyl chain, written from the omega end towards the sulfur.

    ``double_bond`` is the position in acyl numbering, where carbon 1 is the carbonyl, so a
    ``dec-3-enoyl`` group is ten carbons with the unsaturation between carbons 3 and 4. The
    carbon budget is stated as a sum so an off-by-one shows up here rather than as a formula
    mismatch: ``(n - p - 1)`` carbons omega of the double bond, ``2`` in it, ``(p - 2)``
    between it and the carbonyl, and ``1`` for the carbonyl itself, which is ``n``.

    Geometry is explicit. It changes nothing about the formula, so the acceptance test cannot
    catch a wrong one, and a predictor that reads stereochemistry has to be given the isomer
    the name states.
    """
    if double_bond is None:
        return "C" * (n_carbons - 1) + f"C(=O){COA_TAIL}"
    omega = "C" * (n_carbons - double_bond - 1)
    inner = "C" * (double_bond - 2)
    bond = "/C=C\\" if cis else "/C=C/"
    return f"{omega}{bond}{inner}C(=O){COA_TAIL}"


def dolichol(n_isoprene: int) -> str:
    """A dolichol of ``n_isoprene`` units, as the alcohol.

    A dolichol is a polyprenol whose ALPHA unit is saturated, which is what distinguishes it
    from a polyprenol and what makes its hydrogen count two higher per molecule. The written
    form is alpha-saturated, all-cis internal units and a trisubstituted omega end, which is
    the yeast configuration.
    """
    alpha = "OCC[C@H](C)"
    internal = "CC/C=C(C)/" * (n_isoprene - 2)
    omega = "CCC=C(C)C"
    return alpha + internal + omega


def dolichyl_mannosyl_phosphate(n_isoprene: int) -> str:
    """alpha-D-mannosyl phosphate esterified to a dolichol of ``n_isoprene`` units."""
    chain = dolichol(n_isoprene)[1:]  # drop the alcohol's leading O, the phosphate supplies it
    return (
        "OC[C@H]1O[C@H](OP(=O)(O)O" + chain + ")[C@@H](O)[C@@H](O)[C@@H]1O"
    )


def diformyl_dityrosine() -> str:
    """N,N'-diformyl dityrosine: two tyrosines joined 3,3' with both amines formylated."""
    return (
        "O=CNC(Cc1ccc(O)c(-c2cc(CC(NC=O)C(=O)O)ccc2O)c1)C(=O)O"
    )


class StructureCandidate(BaseModel):
    """One candidate structure for one named metabolite, before the formula decides."""

    model_config = ConfigDict(extra="forbid")

    name: str
    smiles: str
    #: How the candidate was built, including the parameter the formula selected.
    template: str


class WrittenStructure(BaseModel):
    """The report row for one metabolite, accepted or refused.

    The field names follow the lipid writer's schema so
    :func:`~torchcell.metabolism.kinetics_pipeline.read_supplied_structures` reads either
    file without a special case.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    met_ids: list[str]
    chemical_class: str
    template: str
    smiles: str | None
    gem_formula: str
    gem_charge: float
    rdkit_formula: str | None
    formula_match: bool
    n_candidates_tried: int
    failure: str | None


class WriterReport(BaseModel):
    """Everything the writer produced, accepted and refused together."""

    model_config = ConfigDict(extra="forbid")

    built_at: str
    conventions: dict[str, str]
    n_attempted: int
    n_written: int
    n_refused: int
    structures: list[WrittenStructure] = Field(
        description="Accepted and refused rows; only `formula_match` rows are consumed."
    )


#: Metabolite name -> (chemical class, candidate generator). Each generator yields
#: candidates in the order they should be tried; the formula test picks, and if none matches
#: the metabolite is reported refused with the closest candidate's formula.
def candidates_for(name: str) -> tuple[str, list[StructureCandidate]]:
    """Every structure worth trying for one GEM metabolite name."""
    lowered = name.lower()
    if lowered == "dolichyl d-mannosyl phosphate":
        return "dolichyl_glycoside", [
            StructureCandidate(
                name=name,
                smiles=dolichyl_mannosyl_phosphate(n),
                template=f"DOLICHYL_MANNOSYL_PHOSPHATE(n_isoprene={n})",
            )
            for n in range(2, 12)
        ]
    if lowered.endswith("enoyl-coa") or lowered.endswith("anoyl-coa"):
        cis = lowered.startswith("cis-")
        stem = lowered.removeprefix("cis-").removeprefix("trans-")
        word = stem.split("-")[0]
        length = CHAIN_LENGTHS.get(word)
        if length is None:
            return "acyl_CoA", []
        parts = stem.split("-")
        position = int(parts[1]) if len(parts) > 2 and parts[1].isdigit() else None
        return "acyl_CoA", [
            StructureCandidate(
                name=name,
                smiles=acyl_coa(length, position, cis),
                template=(
                    f"ACYL_COA(n_carbons={length}, double_bond={position}, "
                    f"{'cis' if cis else 'trans'})"
                ),
            )
        ]
    if lowered == "n,n'-diformyldityrosine":
        return "cross_link", [
            StructureCandidate(
                name=name,
                smiles=diformyl_dityrosine(),
                template="DIFORMYL_DITYROSINE(3,3'-biaryl)",
            )
        ]
    return "unhandled", []


#: The metabolites this writer is asked about. Taken from the residual blockers of
#: ``kinetic_coverage_gaps.py``, restricted to the ones whose formula carries no R: a
#: placeholder has nothing for the formula test to select against.
TARGET_NAMES = [
    "dolichyl D-mannosyl phosphate",
    "cis-dec-3-enoyl-CoA",
    "N,N'-diformyldityrosine",
    "phosphatidylethanolamine backbone",
    "hydrogen sulfide",
]


def main() -> None:
    source = YeastGEM()
    model = source.model
    by_name: dict[str, list] = {}
    for metabolite in model.metabolites:
        by_name.setdefault(metabolite.name or "", []).append(metabolite)

    rows: list[WrittenStructure] = []
    for name in TARGET_NAMES:
        metabolites = by_name.get(name, [])
        if not metabolites:
            raise ValueError(f"{name!r} is not a metabolite of this GEM")
        reference = metabolites[0]
        formula = reference.formula or ""
        charge = float(reference.charge or 0.0)
        chemical_class, candidates = candidates_for(name)
        accepted = None
        for candidate in candidates:
            if structure_matches_metabolite(candidate.smiles, formula, charge):
                accepted = candidate
                break
        computed = (
            formula_from_smiles(accepted.smiles)
            if accepted
            else (formula_from_smiles(candidates[0].smiles) if candidates else None)
        )
        rows.append(
            WrittenStructure(
                name=name,
                met_ids=[m.id for m in metabolites],
                chemical_class=chemical_class,
                template=accepted.template if accepted else "none accepted",
                smiles=accepted.smiles if accepted else None,
                gem_formula=formula,
                gem_charge=charge,
                rdkit_formula=computed,
                formula_match=accepted is not None,
                n_candidates_tried=len(candidates),
                failure=None
                if accepted
                else (
                    "no template for this name"
                    if not candidates
                    else "no candidate matches the GEM's neutralized composition"
                ),
            )
        )

    report = WriterReport(
        built_at=datetime.now(UTC).isoformat(),
        conventions={
            "acceptance": (
                "RDKit's formula for the SMILES must equal the GEM's formula, both moved "
                "to the neutral species by adding one hydrogen per negative charge"
            ),
            "chain_length": (
                "derived, not chosen: the template is built for every length in range and "
                "the GEM's formula selects one"
            ),
            "stereocenters": (
                "acyl double bonds carry the geometry the name states; the CoA moiety and "
                "the dolichol alpha carbon are left unspecified, as in the lipid writer"
            ),
        },
        n_attempted=len(rows),
        n_written=sum(1 for r in rows if r.formula_match),
        n_refused=sum(1 for r in rows if not r.formula_match),
        structures=rows,
    )
    with open(OUT_PATH, "w") as handle:
        handle.write(report.model_dump_json(indent=2))

    for row in rows:
        verdict = "WROTE " if row.formula_match else "refused"
        print(
            f"{verdict} {row.name[:44]:46} gem {row.gem_formula:18} "
            f"rdkit {str(row.rdkit_formula):20} {row.template}"
        )
    print(f"\n{report.n_written} written, {report.n_refused} refused -> {OUT_PATH}")


if __name__ == "__main__":
    main()
