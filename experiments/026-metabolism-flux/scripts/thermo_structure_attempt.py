# experiments/026-metabolism-flux/scripts/thermo_structure_attempt.py
# [[experiments.026-metabolism-flux.scripts.thermo_structure_attempt]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/026-metabolism-flux/scripts/thermo_structure_attempt.py

r"""Take every remaining structureless species as far as a structure can take it, then stop.

After the accession sweep, the ion table and the repeat-unit surrogate, what is left in
``no_structure_available`` is two kinds of thing, and they need separating by evidence
rather than by eye.

**Metalloporphyrins and cytochromes with a definite formula.** heme a, heme o, siroheme and
ferroheme b all HAVE a structure: the compound cache holds one for each, reached through the
GEM's own ChEBI or BiGG accession, and each of those cache compounds even carries a group
vector. What they do not have is a formation energy, and the question this script answers is
whether that is a lookup failure that creating the compound fresh would fix, or a
decomposition failure that nothing fixes. The answer is recorded per compound.

**Protein-linked residues.** ``[protein]-L-lysine``, ``[cytochrome c]-L-lysine`` and
Apocytochrome c carry a definite-looking formula, and their cache structures carry an
ATTACHMENT POINT: ``[NH3+]CCCC[C@H](N-*)C(-*)=O``. A ``*`` is a placeholder for the rest of
the protein, so the species is a residue in a macromolecule rather than a molecule, and no
structure, pKa ladder or decomposition will ever price it. That is detected from the
structure itself rather than from the name, and it moves the species into the permanently
closed class with its own reason.

The composition of any structure used is checked against the GEM's formula before anything
is attempted, on heavy atoms, so a namespace pointing at a different molecule is rejected
rather than priced.

Writes ``results/thermo_structure_attempt.json``.

Run under the equilibrator-ext environment, which carries the compound bridge::

    ~/miniconda3/envs/equilibrator-ext/bin/python \
        experiments/026-metabolism-flux/scripts/thermo_structure_attempt.py \
        --local-cache $DATA_ROOT/tmp/thermo_structure_attempt.sqlite \
        --seed-cache ~/.cache/equilibrator/compounds.sqlite
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import re
from datetime import datetime, timezone

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
RESULTS = osp.join(EXPERIMENT_ROOT, "026-metabolism-flux", "results")
PIPELINE_JSON = osp.join(RESULTS, "thermo_pipeline_yeast9.json")
AUDIT_CSV = osp.join(RESULTS, "unresolved_metabolites.csv")
LADDER_MOLGPKA = osp.join(RESULTS, "external_pka_apply_ladders_molgpka.json")
OUT_PATH = osp.join(RESULTS, "thermo_structure_attempt.json")

P_H = 7.2
IONIC_STRENGTH_M = 0.25
TEMPERATURE_K = 303.15
P_MG = 3.0

#: An attachment point in a structure. A species whose structure carries one is a residue in
#: a macromolecule, not a molecule, so it can never carry a formation energy of its own.
ATTACHMENT_POINT = "*"

_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*)")


def heavy_composition(formula: str) -> dict[str, int]:
    """Heavy-atom composition, hydrogens dropped; the cache's proton count is pH dependent."""
    out: dict[str, int] = {}
    for element, count in _TOKEN.findall(formula or ""):
        if not element or element == "H":
            continue
        out[element] = out.get(element, 0) + (int(count) if count else 1)
    return out


class StructureAttempt(BaseModel):
    """One species taken as far as a structure goes, with the reason it stopped."""

    met_id: str
    met_ids: list[str] = Field(default_factory=list)
    name: str
    formula: str
    charge: int
    n_reactions_blocked: int
    accession_used: str | None = None
    cache_compound_id: int | None = None
    structure: str | None = None
    structure_has_attachment_point: bool = False
    composition_matches: bool | None = None
    cache_has_group_vector: bool | None = None
    created_compound_id: int | None = None
    created_group_decomposition_ok: bool | None = None
    dgf_prime_kj_per_mol: float | None = None
    outcome: str = ""
    reason: str = ""
    remedy: str = ""


class StructureAttemptReport(BaseModel):
    """What a structure could and could not buy for the species that still have none."""

    built_at: str
    conditions: dict[str, float]
    composition_test: str = (
        "heavy-atom composition of the cache structure against the GEM formula; hydrogens "
        "are excluded because the cache stores a microspecies whose proton count depends on "
        "pH while the GEM's depends on its own charge convention"
    )
    n_candidates: int
    n_priced: int
    n_attachment_point: int
    n_decomposition_failed: int
    outcomes: dict[str, int] = Field(default_factory=dict)
    records: list[StructureAttempt] = Field(default_factory=list)


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()  # noqa: UP017


def main() -> None:
    import shutil

    import pandas as pd
    from equilibrator_api import Q_, ComponentContribution
    from equilibrator_assets.local_compound_cache import LocalCompoundCache

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-cache", required=True)
    parser.add_argument("--seed-cache", default=None)
    args = parser.parse_args()

    with open(PIPELINE_JSON) as handle:
        pipeline = json.load(handle)
    candidates = {
        row["met_id"]
        for row in pipeline["metabolites"]
        if row["dgf_prime_kj_per_mol"] is None
        and str(row.get("reason")) == "no_structure_available"
    }
    blocked: dict[str, int] = {}
    for reaction in pipeline["reactions"]:
        if reaction["delta_r_g0_kj_per_mol"] is not None:
            continue
        for met_id in reaction["blocking_met_ids"]:
            blocked[met_id] = blocked.get(met_id, 0) + 1
    audit = pd.read_csv(AUDIT_CSV).set_index("met_id")

    if args.seed_cache and not osp.isfile(args.local_cache):
        os.makedirs(osp.dirname(args.local_cache), exist_ok=True)
        shutil.copy2(args.seed_cache, args.local_cache)

    cc = ComponentContribution()
    cc.p_h = Q_(P_H)
    cc.ionic_strength = Q_(f"{IONIC_STRENGTH_M}M")
    cc.temperature = Q_(f"{TEMPERATURE_K}K")
    cc.p_mg = Q_(P_MG)
    local = LocalCompoundCache()
    local.load_cache(args.local_cache)

    records: list[StructureAttempt] = []
    by_name: dict[str, StructureAttempt] = {}
    for met_id in sorted(candidates):
        if met_id not in audit.index:
            continue
        row = audit.loc[met_id]
        record = StructureAttempt(
            met_id=met_id,
            met_ids=[met_id],
            name=str(row["name"]),
            formula=str(row["formula"]) if pd.notna(row["formula"]) else "",
            charge=int(row["charge"]) if pd.notna(row["charge"]) else 0,
            n_reactions_blocked=blocked.get(met_id, 0),
        )
        wanted = heavy_composition(record.formula)
        annotations = str(row["annotations"]) if pd.notna(row["annotations"]) else ""

        # 1. the best structure any of the GEM's own accessions reaches
        for token in annotations.split(";"):
            if "=" not in token:
                continue
            namespace, _, accession = token.partition("=")
            try:
                compound = cc.ccache.get_compound(f"{namespace}:{accession}")
            except Exception:
                continue
            if compound is None or not compound.smiles:
                continue
            record.accession_used = f"{namespace}:{accession}"
            record.cache_compound_id = int(compound.id)
            record.structure = str(compound.smiles)
            record.cache_has_group_vector = compound.group_vector is not None
            record.composition_matches = (
                heavy_composition(compound.formula or "") == wanted
            )
            break

        if record.structure is None:
            record.outcome = "no_structure_in_any_namespace"
            record.reason = "no_structure_available"
            record.remedy = "a structure from a curator; no namespace the GEM cites has one"
            records.append(record)
            continue

        # 2. an attachment point means the species is not a molecule
        if ATTACHMENT_POINT in record.structure:
            record.structure_has_attachment_point = True
            record.outcome = "protein_linked_residue"
            record.reason = "placeholder_structure"
            record.remedy = (
                "nothing is possible: the structure carries an attachment point, so the "
                "species is a residue inside a macromolecule rather than a molecule, and "
                "no structure, ladder or decomposition gives a residue a formation energy "
                "of its own"
            )
            records.append(record)
            print(f"  {met_id} {record.name[:34]:34s} {record.outcome}", flush=True)
            continue

        if not record.composition_matches:
            record.outcome = "composition_mismatch"
            record.reason = "no_structure_available"
            record.remedy = (
                "the only structure the GEM's accessions reach is a different molecule; "
                "pricing it would be worse than leaving the reaction unpriced"
            )
            records.append(record)
            print(f"  {met_id} {record.name[:34]:34s} {record.outcome}", flush=True)
            continue

        # 3. create the compound FRESH from that structure, which is the step a lookup
        #    cannot do, and see whether the decomposer accepts it
        try:
            results = local.get_compounds(
                [record.structure], mol_format="smiles", bypass_chemaxon=True
            )
        except Exception as exc:
            record.outcome = "compound_creation_raised"
            record.reason = "group_decomposition_failed"
            record.remedy = f"{type(exc).__name__}: {exc}"[:200]
            records.append(record)
            print(f"  {met_id} {record.name[:34]:34s} {record.outcome}", flush=True)
            continue
        created = results[0].compound if results else None
        record.created_compound_id = int(created.id) if created is not None else None
        record.created_group_decomposition_ok = (
            created is not None and created.group_vector is not None
        )
        if created is None or not record.created_group_decomposition_ok:
            record.outcome = "failed_group_decomposition"
            record.reason = "group_decomposition_failed"
            record.remedy = (
                "component contribution has no group decomposition for this structure, so "
                "a pKa ladder and a structure both change nothing; the remedy is a "
                "decomposition or a measured value, not more retrieval"
            )
            records.append(record)
            print(f"  {met_id} {record.name[:34]:34s} {record.outcome}", flush=True)
            continue
        mu, _sigma, _ = cc.standard_dg_formation(created)
        if mu is None:
            record.outcome = "decomposed_but_no_formation_energy"
            record.reason = "group_decomposition_failed"
            record.remedy = (
                "the structure decomposes and component contribution still returns no "
                "formation energy, which happens where a group in the decomposition has no "
                "fitted contribution; a measured value is the only remedy"
            )
            records.append(record)
            print(f"  {met_id} {record.name[:34]:34s} {record.outcome}", flush=True)
            continue
        correction = created.transform(
            Q_(P_H), Q_(f"{IONIC_STRENGTH_M}M"), Q_(f"{TEMPERATURE_K}K"), Q_(P_MG)
        )
        record.dgf_prime_kj_per_mol = float(mu) + float(correction.m_as("kJ/mol"))
        record.outcome = "priced"
        record.reason = ""
        record.remedy = ""
        records.append(record)
        by_name[record.name] = record
        print(
            f"  {met_id} {record.name[:34]:34s} priced {record.dgf_prime_kj_per_mol:.2f}",
            flush=True,
        )

    for met_id in sorted(candidates):
        if met_id in audit.index:
            name = str(audit.loc[met_id, "name"])
            if name in by_name and met_id not in by_name[name].met_ids:
                by_name[name].met_ids.append(met_id)

    outcomes: dict[str, int] = {}
    for record in records:
        outcomes[record.outcome] = outcomes.get(record.outcome, 0) + 1
    report = StructureAttemptReport(
        built_at=_timestamp(),
        conditions={
            "p_h": P_H,
            "ionic_strength_m": IONIC_STRENGTH_M,
            "temperature_k": TEMPERATURE_K,
            "p_mg": P_MG,
        },
        n_candidates=len(records),
        n_priced=sum(1 for r in records if r.outcome == "priced"),
        n_attachment_point=sum(1 for r in records if r.structure_has_attachment_point),
        n_decomposition_failed=sum(
            1 for r in records if r.reason == "group_decomposition_failed"
        ),
        outcomes=dict(sorted(outcomes.items(), key=lambda kv: -kv[1])),
        records=records,
    )
    with open(OUT_PATH, "w") as handle:
        handle.write(report.model_dump_json(indent=1))
    print(f"wrote {OUT_PATH}")
    print(f"  {report.outcomes}")


if __name__ == "__main__":
    main()
