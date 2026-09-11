# experiments/026-metabolism-flux/scripts/thermo_curation_price.py
# [[experiments.026-metabolism-flux.scripts.thermo_curation_price]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/026-metabolism-flux/scripts/thermo_curation_price.py

r"""Run the structure route on every curated structure, under the compound bridge.

The GEM driver and component contribution cannot share a process: ``cobra`` is not in the
``equilibrator-ext`` environment and ``equilibrator_api`` is not in the torchcell one. This
script is the second half of that handshake. ``thermo_pipeline_yeast9.py --curation-dir``
writes ``results/thermo_curation_pricing_requests.json``, one entry per (curated structure,
pKa ladder, compartment conditions) an ADMITTED record implies; this script prices each of
them and writes ``results/thermo_curation_priced_structures.json``, which the driver then
replays through
:class:`~torchcell.metabolism.thermo_curation.PricedStructureTable`.

Both files are derived from the same code
(:func:`~torchcell.metabolism.thermo_curation.pricing_requests` and
:class:`~torchcell.metabolism.thermo_curation.EquilibratorStructurePricer`), so the request
and the answer cannot drift apart, and a request the artifact does not answer RAISES in the
driver rather than being filled with something else.

It also re-verifies the Legendre transform this repository restates in
:func:`~torchcell.metabolism.thermo_pipeline.legendre_transform_kj` against
``equilibrator_cache.thermodynamic_constants._legendre_transform`` over a grid, and records
the worst deviation in the output, the same way ``fetch_ion_formation_energies.py`` verifies
the Debye-Huckel restatement. The restatement is used on every curated formation energy, so
it is checked on every rebuild rather than trusted.

Run from the repository root::

    PYTHONPATH=$PWD ~/miniconda3/envs/equilibrator-ext/bin/python \
        experiments/026-metabolism-flux/scripts/thermo_curation_price.py \
        --local-cache $DATA_ROOT/tmp/thermo_curation.sqlite \
        --seed-cache ~/.cache/equilibrator/compounds.sqlite
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import shutil
from datetime import datetime, timezone

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field

from torchcell.metabolism.thermo_curation import (
    ConservedMoietyRequest,
    EquilibratorStructurePricer,
    PricedMoietyReaction,
    PricedStructure,
    PricingRequest,
    requests_digest,
)
from torchcell.metabolism.thermo_pipeline import legendre_transform_kj

load_dotenv()

DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
RESULTS = osp.join(EXPERIMENT_ROOT, "026-metabolism-flux", "results")
REQUESTS_PATH = osp.join(RESULTS, "thermo_curation_pricing_requests.json")
OUT_PATH = osp.join(RESULTS, "thermo_curation_priced_structures.json")
MOIETY_REQUESTS_PATH = osp.join(RESULTS, "thermo_moiety_requests.json")
MOIETY_OUT_PATH = osp.join(RESULTS, "thermo_moiety_priced.json")

#: A group whose count is below this is treated as absent from a difference vector.
GROUP_TOLERANCE = 1e-9

#: Above this the infinite-variance component is nonzero and the estimate is not usable.
SIGMA_INF_TOLERANCE = 1e-8


class PricedStructureSet(BaseModel):
    """Every curated structure the route was run on, and what it returned."""

    built_at: str
    requests_path: str
    requests_sha256: str
    #: A content hash of the request SET. The driver refuses an artifact whose digest is
    #: not the one its own request list hashes to, which is the only thing that stops a
    #: coverage number from being read off answers to an older question.
    requests_digest: str
    local_cache_path: str
    n_requests: int
    n_priced: int
    outcomes: dict[str, int] = Field(default_factory=dict)
    #: Worst |our transform - eQuilibrator's| over the verification grid, in kJ/mol.
    legendre_transform_max_deviation_kj_per_mol: float
    records: list[PricedStructure] = Field(default_factory=list)


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()  # noqa: UP017


def _sha256(path: str) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def verify_legendre_transform() -> float:
    """Worst deviation of our restated transform from eQuilibrator's, over a grid.

    The grid spans the compartment pHs the 026 build uses, the two ionic strengths and
    temperatures in play, and proton counts up to twelve, which is where the difference
    between the CODATA gas constant and eQuilibrator's rounded one is largest.
    """
    from equilibrator_cache.thermodynamic_constants import (
        _legendre_transform,
        default_R_in_kJ_per_mol_per_K,
    )

    worst = 0.0
    for p_h in (5.0, 6.2, 6.6, 7.0, 7.1, 7.2, 7.8):
        for ionic in (0.0, 0.1, 0.25):
            for temperature in (298.15, 303.15):
                for charge in (-8, -4, -2, 0, 1, 3):
                    for protons in (0, 1, 5, 12):
                        reference = (
                            default_R_in_kJ_per_mol_per_K
                            * temperature
                            * _legendre_transform(
                                pH=p_h,
                                pMg=3.0,
                                ionic_strength_M=ionic,
                                T_in_K=temperature,
                                charge=charge,
                                num_protons=protons,
                                num_magnesiums=0,
                            )
                        )
                        ours = legendre_transform_kj(
                            charge=charge,
                            num_protons=protons,
                            p_h=p_h,
                            ionic_strength_m=ionic,
                            temperature_k=temperature,
                        )
                        worst = max(worst, abs(reference - ours))
    return worst


class MoietyPricedSet(BaseModel):
    """Every reaction the conserved-moiety route was run on, and what it returned."""

    built_at: str
    requests_path: str
    requests_sha256: str
    n_requests: int
    n_priced: int
    outcomes: dict[str, int] = Field(default_factory=dict)
    records: list[PricedMoietyReaction] = Field(default_factory=list)


def _element_symbol(atomic_number: int) -> str:
    """Element symbol for an atomic number, through OpenBabel's own table."""
    from openbabel import openbabel

    return str(openbabel.GetSymbol(atomic_number))


def _heavy_composition_nonempty(formula: str) -> bool:
    """Whether this formula has any heavy atom at all. A bare proton has none."""
    from torchcell.metabolism.thermo_curation import _heavy_composition

    return bool(_heavy_composition(formula))


def _trusted_cache_structure(compound: object, formula: str) -> bool:
    """Whether a cache row's own composition agrees with the GEM's formula on heavy atoms.

    The compound cache resolves by InChIKey and hands back whatever row it already holds.
    Several of those rows are stale: heme b's carries no iron, siroheme's and
    4-nitrophenol's atom bags disagree with their own InChI. Pricing a reaction off a
    structure that is missing the metal is not an approximation, it is a different
    reaction, so a row whose heavy-atom composition disagrees with the GEM is refused.
    """
    from torchcell.metabolism.thermo_curation import _heavy_composition

    wanted = _heavy_composition(formula)
    if wanted is None:
        return False
    if not wanted:
        # A bare proton has no heavy atoms to compare, and the transformed convention
        # prices it at zero anyway. There is nothing here for a stale row to get wrong.
        return True
    atom_bag = getattr(compound, "atom_bag", None)
    if not atom_bag:
        return False
    found = {k: float(v) for k, v in atom_bag.items() if k not in ("H", "e-")}
    return found == wanted


def price_moieties(args: argparse.Namespace) -> None:
    """Run the conserved-moiety route on every requested reaction."""
    import collections

    import numpy as np
    from equilibrator_api import Q_, ComponentContribution
    from equilibrator_assets.group_decompose import GroupDecomposer
    from equilibrator_assets.molecule import Molecule

    with open(args.moiety_requests) as handle:
        payload = json.load(handle)
    requests = [
        ConservedMoietyRequest.model_validate(row) for row in payload["requests"]
    ]
    print(f"{len(requests)} conserved-moiety requests from {args.moiety_requests}")

    cc = ComponentContribution()
    pre = cc.predictor.preprocess
    decomposer = GroupDecomposer()
    group_names = list(cc.predictor.params.train_G.columns)
    training = cc.predictor.params.train_G.values
    coverage = (np.abs(training[:, : pre.Ng]) > 0).sum(axis=0)
    # A protonated group nobody trained on, whose conjugate base IS trained on, is a
    # representation of the same site rather than new chemistry. Moving the count to the
    # base is the convention the rest of the table already uses, because the compound
    # cache stores every compound at its pH 7 major microspecies.
    conjugate: dict[int, int] = {}
    for i, name in enumerate(group_names[: pre.Ng]):
        if "[H1 Z0" not in name or coverage[i] > 0:
            continue
        base = name.replace("[H1 Z0", "[H0 Z-1")
        if base in group_names[: pre.Ng]:
            j = group_names.index(base)
            if coverage[j] > 0:
                conjugate[i] = j

    records: list[PricedMoietyReaction] = []
    for request in requests:
        row = PricedMoietyReaction(reaction_id=request.reaction_id)
        x = np.zeros(pre.Nc + pre.Ng)
        unassigned: collections.Counter = collections.Counter()
        transform = 0.0
        failed = ""
        for part in request.participants:
            compound = None
            for namespace, values in sorted(part.accessions.items()):
                for value in values:
                    try:
                        compound = cc.ccache.get_compound(f"{namespace}:{value}")
                    except Exception:  # noqa: BLE001 - the cache raises many types
                        compound = None
                    if compound is not None:
                        break
                if compound is not None:
                    break
            if not _heavy_composition_nonempty(part.formula):
                # A bare proton is excluded from the reaction vector, exactly as
                # eQuilibrator's own Reaction.items(protons=False) excludes it: in the
                # transformed convention its formation energy is zero at every pH.
                row.participants[part.met_id] = "proton_excluded"
                continue
            vector = pre.get_compound_vector(compound) if compound is not None else None
            trusted = compound is not None and _trusted_cache_structure(
                compound, part.formula
            )
            if vector is not None and trusted:
                x += part.coefficient * vector
                row.participants[part.met_id] = (
                    "reactant_contribution"
                    if compound.id in pre._compound_ids
                    else "group_contribution"
                )
                mu_i, _, _ = pre.get_compound_prediction(compound)
                transform += part.coefficient * float(
                    compound.transform(
                        Q_(part.p_h),
                        Q_(f"{part.ionic_strength_m}M"),
                        Q_(f"{part.temperature_k}K"),
                        Q_(part.p_mg),
                    ).m_as("kJ/mol")
                )
                continue
            smiles = part.smiles or (
                compound.smiles if compound is not None and trusted else None
            )
            if smiles is None:
                row.participants[part.met_id] = (
                    "no_trusted_structure" if compound is not None else "no_structure"
                )
                failed = failed or (
                    "untrusted_structure" if compound is not None else "no_structure"
                )
                continue
            decomposition = decomposer.Decompose(
                Molecule.FromSmiles(smiles),
                ignore_protonations=False,
                raise_exception=False,
            )
            vec = np.array(list(decomposition.AsVector()), dtype=float)
            moved = 0.0
            for i, j in conjugate.items():
                if abs(vec[i]) > GROUP_TOLERANCE:
                    moved += vec[i]
                    row.protonation_normalized[group_names[i]] = (
                        row.protonation_normalized.get(group_names[i], 0.0)
                        + part.coefficient * vec[i]
                    )
                    vec[j] += vec[i]
                    vec[i] = 0.0
            x[pre.Nc :] += part.coefficient * vec
            for atom, _index in decomposition.UnassignedAtoms():
                symbol = _element_symbol(atom.GetAtomicNum())
                unassigned[symbol] += part.coefficient
            protons = float(decomposition.Hydrogens()) - moved
            charge = float(decomposition.NetCharge()) - moved
            transform += part.coefficient * legendre_transform_kj(
                charge=int(charge),
                num_protons=protons,
                p_h=part.p_h,
                ionic_strength_m=part.ionic_strength_m,
                temperature_k=part.temperature_k,
            )
            row.participants[part.met_id] = "lenient_decomposition"
        row.unassigned_residual = {
            element: count
            for element, count in sorted(unassigned.items())
            if abs(count) > GROUP_TOLERANCE
        }
        row.difference_groups = [
            {
                "group": group_names[j],
                "difference": float(x[pre.Nc + j]),
                "training_compounds": int(coverage[j]),
            }
            for j in range(pre.Ng)
            if abs(x[pre.Nc + j]) > GROUP_TOLERANCE
        ]
        if failed:
            row.outcome = failed
            row.detail = "a participant has no structure this route can trust"
        elif row.unassigned_residual:
            row.outcome = "moiety_does_not_cancel"
            row.detail = (
                "the atoms no group covers do not match across the two sides: "
                f"{row.unassigned_residual}"
            )
        else:
            sigma_inf = float(np.linalg.norm(pre.L_inf @ x))
            row.sigma_inf_norm = sigma_inf
            if sigma_inf > SIGMA_INF_TOLERANCE:
                row.outcome = "group_in_null_space"
                zero = [
                    g["group"]
                    for g in row.difference_groups
                    if g["training_compounds"] == 0
                ]
                row.detail = (
                    "the group difference has an infinite-variance component; groups no "
                    f"training compound carries: {zero or 'none named'}"
                )
            else:
                row.standard_kj_per_mol = float(pre.mu @ x)
                row.transform_kj_per_mol = transform
                row.delta_r_g0_kj_per_mol = row.standard_kj_per_mol + transform
                row.sigma_kj_per_mol = float(np.linalg.norm(pre.L_c @ x))
                row.outcome = "priced"
        records.append(row)
        value = (
            f"{row.delta_r_g0_kj_per_mol:10.3f}"
            if row.delta_r_g0_kj_per_mol is not None
            else "         -"
        )
        print(f"  {request.reaction_id:10s} {value}  {row.outcome}", flush=True)

    outcomes: dict[str, int] = {}
    for record in records:
        outcomes[record.outcome] = outcomes.get(record.outcome, 0) + 1
    result = MoietyPricedSet(
        built_at=_timestamp(),
        requests_path=args.moiety_requests,
        requests_sha256=_sha256(args.moiety_requests),
        n_requests=len(requests),
        n_priced=sum(1 for r in records if r.priced),
        outcomes=dict(sorted(outcomes.items(), key=lambda kv: -kv[1])),
        records=records,
    )
    with open(args.moiety_out, "w") as handle:
        handle.write(result.model_dump_json(indent=1))
    print(f"wrote {args.moiety_out}: {result.n_priced}/{result.n_requests} priced")
    print(f"  outcomes: {result.outcomes}")


class AccessionCompositionRow(BaseModel):
    """One accession-priced metabolite, and whether its structure is the GEM's molecule."""

    model_config = ConfigDict(extra="forbid")

    met_id: str
    name: str = ""
    gem_formula: str = ""
    gem_charge: int = 0
    accession_used: str = ""
    compound_id: int | None = None
    structure_formula: str = ""
    #: ``match``, ``repeat_unit_multiple``, ``other`` or ``no_compound``.
    classification: str = ""
    #: How many repeat units the structure carries beyond the GEM's formula.
    repeat_unit_offset: float | None = None
    #: What the pipeline currently prices this metabolite at.
    dgf_prime_kj_per_mol: float | None = None
    detail: str = ""


class AccessionCompositionReport(BaseModel):
    """The composition sweep over every accession-priced metabolite."""

    built_at: str
    repeat_unit: str
    n_checked: int
    n_match: int
    n_repeat_unit_multiple: int
    n_other: int
    #: The repeat-unit offsets seen, and how many metabolites carry each. A family is
    #: consistent when this has exactly one entry.
    repeat_unit_offsets: dict[str, int] = Field(default_factory=dict)
    family_consistent: bool = False
    rows: list[AccessionCompositionRow] = Field(default_factory=list)
    dimer_differences: list[dict[str, object]] = Field(default_factory=list)
    note: str = (
        "The accession route had NO composition check: it accepted the first accession "
        "that resolved to a compound carrying a formation energy, whatever molecule that "
        "compound is. Every row classified other is a price for a different molecule."
    )


def _formula_difference(
    structure: dict[str, float], gem: dict[str, float]
) -> dict[str, float]:
    """Structure minus GEM, element by element, keeping only what differs."""
    out: dict[str, float] = {}
    for element in set(structure) | set(gem):
        delta = structure.get(element, 0.0) - gem.get(element, 0.0)
        if abs(delta) > GROUP_TOLERANCE:
            out[element] = delta
    return out


def _repeat_unit_multiple(
    difference: dict[str, float], repeat: dict[str, float]
) -> float | None:
    """How many whole repeat units the difference is, or ``None`` if it is not a multiple."""
    if not difference:
        return 0.0
    ratios = set()
    for element, delta in difference.items():
        if element not in repeat:
            return None
        ratio = delta / repeat[element]
        if abs(ratio - round(ratio)) > 1e-9:
            return None
        ratios.add(round(ratio))
    for element in repeat:
        if element not in difference:
            return None
    return float(ratios.pop()) if len(ratios) == 1 else None


def sweep_accessions(args: argparse.Namespace) -> None:
    """Check every accession-priced metabolite's structure against the GEM's formula."""
    import numpy as np
    from equilibrator_api import Q_, ComponentContribution

    from torchcell.metabolism.thermo_curation import _heavy_composition

    with open(args.accession_requests) as handle:
        payload = json.load(handle)
    rows_in = payload["metabolites"]
    repeat = _heavy_composition(args.repeat_unit) or {}
    print(
        f"{len(rows_in)} accession-priced metabolites, repeat unit {args.repeat_unit}"
    )

    cc = ComponentContribution()
    rows: list[AccessionCompositionRow] = []
    for entry in rows_in:
        row = AccessionCompositionRow(
            met_id=entry["met_id"],
            name=entry.get("name", ""),
            gem_formula=entry.get("formula", ""),
            gem_charge=int(entry.get("charge", 0)),
            dgf_prime_kj_per_mol=entry.get("dgf_prime_kj_per_mol"),
        )
        compound = None
        for namespace, values in sorted((entry.get("accessions") or {}).items()):
            for value in values:
                try:
                    compound = cc.ccache.get_compound(f"{namespace}:{value}")
                except Exception:  # noqa: BLE001 - the cache raises many types
                    compound = None
                if compound is not None and compound.formula:
                    row.accession_used = f"{namespace}:{value}"
                    break
            if row.accession_used:
                break
        if compound is None or not compound.formula:
            row.classification = "no_compound"
            rows.append(row)
            continue
        row.compound_id = int(compound.id)
        row.structure_formula = str(compound.formula)
        wanted = _heavy_composition(row.gem_formula)
        found = _heavy_composition(row.structure_formula)
        if wanted is None or found is None:
            row.classification = "no_compound"
            row.detail = "one of the two formulas has no definite composition"
            rows.append(row)
            continue
        difference = _formula_difference(found, wanted)
        if not difference:
            row.classification = "match"
        else:
            multiple = _repeat_unit_multiple(difference, repeat)
            if multiple is not None and multiple != 0:
                row.classification = "repeat_unit_multiple"
                row.repeat_unit_offset = multiple
                row.detail = f"structure carries {multiple:+g} x {args.repeat_unit}"
            else:
                row.classification = "other"
                row.detail = f"structure minus GEM = {difference}"
        rows.append(row)

    offsets: dict[str, int] = {}
    for row in rows:
        if row.classification == "repeat_unit_multiple":
            key = f"{row.repeat_unit_offset:+g}"
            offsets[key] = offsets.get(key, 0) + 1

    dimers: list[dict[str, object]] = []
    for spec in args.dimer or []:
        name, accession, monomer_accession, water_accession = spec.split(",")
        values: dict[str, object] = {"name": name, "dimer_accession": accession}
        try:
            dimer = cc.ccache.get_compound(accession)
            water = cc.ccache.get_compound(water_accession)
        except Exception as exc:  # noqa: BLE001
            values["error"] = f"{type(exc).__name__}: {exc}"[:200]
            dimers.append(values)
            continue
        if dimer is None or water is None:
            values["error"] = "the cache has no compound for one of the accessions"
            dimers.append(values)
            continue
        pre = cc.predictor.preprocess
        mu_d, _, inf_d = pre.get_compound_prediction(dimer)
        mu_w, _, inf_w = pre.get_compound_prediction(water)
        if mu_d is None or mu_w is None:
            values["error"] = "no formation energy for the dimer or for water"
            dimers.append(values)
            continue
        values["dimer_compound_id"] = int(dimer.id)
        values["dimer_formula"] = str(dimer.formula)
        values["dimer_estimable"] = bool(
            np.linalg.norm(inf_d) < SIGMA_INF_TOLERANCE
            and np.linalg.norm(inf_w) < SIGMA_INF_TOLERANCE
        )
        for p_h in args.dimer_ph or [7.2]:
            correction_d = float(
                dimer.transform(Q_(p_h), Q_("0.25M"), Q_("303.15K"), Q_(3.0)).m_as(
                    "kJ/mol"
                )
            )
            correction_w = float(
                water.transform(Q_(p_h), Q_("0.25M"), Q_("303.15K"), Q_(3.0)).m_as(
                    "kJ/mol"
                )
            )
            residue = ((mu_d + correction_d) - (mu_w + correction_w)) / 2.0
            values[f"residue_dgf_prime_kj_per_mol_at_ph_{p_h}"] = residue
        dimers.append(values)
        print(f"  dimer difference {name}: {values}")

    report = AccessionCompositionReport(
        built_at=_timestamp(),
        repeat_unit=args.repeat_unit,
        n_checked=len(rows),
        n_match=sum(1 for r in rows if r.classification == "match"),
        n_repeat_unit_multiple=sum(
            1 for r in rows if r.classification == "repeat_unit_multiple"
        ),
        n_other=sum(1 for r in rows if r.classification == "other"),
        repeat_unit_offsets=offsets,
        family_consistent=len(offsets) <= 1,
        rows=rows,
        dimer_differences=dimers,
    )
    with open(args.accession_out, "w") as handle:
        handle.write(report.model_dump_json(indent=1))
    print(f"wrote {args.accession_out}")
    print(
        f"  match {report.n_match}, repeat-unit multiple "
        f"{report.n_repeat_unit_multiple} (offsets {report.repeat_unit_offsets}, "
        f"family consistent {report.family_consistent}), other {report.n_other}"
    )
    for row in rows:
        if row.classification == "other":
            print(
                f"    {row.met_id} {row.name[:30]:32s} {row.gem_formula:16s} against "
                f"{row.structure_formula:16s} {row.detail}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", default=REQUESTS_PATH)
    parser.add_argument("--out", default=OUT_PATH)
    parser.add_argument("--moiety-requests", default=MOIETY_REQUESTS_PATH)
    parser.add_argument("--moiety-out", default=MOIETY_OUT_PATH)
    parser.add_argument(
        "--skip-structures",
        action="store_true",
        help="run only the conserved-moiety half",
    )
    parser.add_argument(
        "--accession-requests",
        default=osp.join(RESULTS, "thermo_accession_check_requests.json"),
    )
    parser.add_argument(
        "--accession-out",
        default=osp.join(RESULTS, "thermo_accession_composition.json"),
    )
    parser.add_argument(
        "--repeat-unit",
        default="C5H8",
        help="the polymer repeat unit a family may legitimately differ by (isoprene)",
    )
    parser.add_argument(
        "--dimer",
        action="append",
        default=[],
        metavar="NAME,DIMER_ACC,MONOMER_ACC,WATER_ACC",
        help="a residue energy to derive as (dimer - water)/2, the mannan convention",
    )
    parser.add_argument("--dimer-ph", action="append", type=float, default=[])
    parser.add_argument(
        "--local-cache",
        required=True,
        help="a WRITABLE sqlite the created compounds are written into",
    )
    parser.add_argument(
        "--seed-cache",
        default=None,
        help="copy this sqlite to --local-cache when it does not exist yet",
    )
    args = parser.parse_args()

    with open(args.requests) as handle:
        payload = json.load(handle)
    requests = [PricingRequest.model_validate(row) for row in payload["requests"]]
    print(f"{len(requests)} pricing requests from {args.requests}")

    deviation = verify_legendre_transform()
    print(f"Legendre transform restatement, worst deviation: {deviation:.3e} kJ/mol")
    if deviation > 1e-9:
        raise ValueError(
            "the restated Legendre transform disagrees with eQuilibrator's by "
            f"{deviation:.3e} kJ/mol; every curated formation energy would inherit that"
        )

    if args.seed_cache and not osp.isfile(args.local_cache):
        os.makedirs(osp.dirname(args.local_cache), exist_ok=True)
        shutil.copy2(args.seed_cache, args.local_cache)
        print(f"seeded {args.local_cache} from {args.seed_cache}")

    pricer = EquilibratorStructurePricer(args.local_cache)
    records: list[PricedStructure] = []
    for request in requests:
        priced = pricer.price(
            request.smiles,
            request.pkas,
            request.p_h,
            request.ionic_strength_m,
            request.temperature_k,
            request.p_mg,
        )
        records.append(priced)
        value = (
            f"{priced.dgf_prime_kj_per_mol:12.3f}"
            if priced.dgf_prime_kj_per_mol is not None
            else "           -"
        )
        print(
            f"  {','.join(request.slugs):26s} pH {request.p_h:4.1f} {value}"
            f"  {priced.outcome}",
            flush=True,
        )

    outcomes: dict[str, int] = {}
    for record in records:
        outcomes[record.outcome] = outcomes.get(record.outcome, 0) + 1
    result = PricedStructureSet(
        built_at=_timestamp(),
        requests_path=args.requests,
        requests_sha256=_sha256(args.requests),
        requests_digest=requests_digest(requests),
        local_cache_path=args.local_cache,
        n_requests=len(requests),
        n_priced=sum(1 for r in records if r.priced),
        outcomes=dict(sorted(outcomes.items(), key=lambda kv: -kv[1])),
        legendre_transform_max_deviation_kj_per_mol=deviation,
        records=records,
    )
    with open(args.out, "w") as handle:
        handle.write(result.model_dump_json(indent=1))
    print(f"wrote {args.out}: {result.n_priced}/{result.n_requests} priced")
    print(f"  outcomes: {result.outcomes}")
    if osp.isfile(args.moiety_requests):
        price_moieties(args)
    else:
        print(f"skip the conserved-moiety route: no {args.moiety_requests}")
    if osp.isfile(args.accession_requests):
        sweep_accessions(args)
    else:
        print(f"skip the accession sweep: no {args.accession_requests}")


if __name__ == "__main__":
    main()
