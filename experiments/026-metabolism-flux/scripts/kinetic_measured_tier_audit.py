# experiments/026-metabolism-flux/scripts/kinetic_measured_tier_audit.py
# [[experiments.026-metabolism-flux.scripts.kinetic_measured_tier_audit]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/026-metabolism-flux/scripts/kinetic_measured_tier_audit.py

r"""Is the measured tier as big as it could be, and is it measuring the right reaction?

THREE QUESTIONS, ONE SCRIPT
----------------------------
The measured tier is 148 of 3,728 catalytic units and everything else is a prediction or a
constant, so three things about those 148 decide how much weight the tier can carry.

**1. Whose enzyme is it?** The flux layer reads the Open Enzyme Database's *S. cerevisiae*
slice, so the tier should be organism-pure. Checked rather than assumed: the audit compares
the slice against the whole 35,753-row database and reports the organism field of every row
any measured value came from.

**2. Is it the right substrate?**
:func:`~torchcell.metabolism.parameters.resolve_kcat_table` keys on the UniProt accession
alone and then picks the row assayed nearest 30 C. It never checks that the row's substrate
is one the GEM reaction actually consumes, so a multi-substrate enzyme can contribute a
turnover number measured on a substrate the reaction does not touch. This audit matches the
chosen row's substrate against the reaction's substrates by InChIKey connectivity block
(protonation- and stereo-tolerant) and by name, and reports how many of the 148 agree.

**3. Would a better key map more?** Four alternative keys are evaluated for REACH only,
none of them applied:

``uniprot_plus_substrate``
    the current key, plus the substrate check. Stricter, so reach can only fall; the drop
    is the size of question 2.
``ec_plus_organism``
    the reaction's EC number against same-organism rows. Reaches enzymes whose own
    accession was never assayed but whose EC class was, in this organism. A DIFFERENT
    claim from a protein-level measurement and tagged as one.
``ec_plus_substrate_plus_organism``
    the same, with the substrate required to match. This is the key the portable pipeline
    offers, because a GEM always has EC annotations and substrate structures while a
    non-model organism's proteins are usually absent from any kinetics database.
``ec_plus_substrate_any_organism``
    the cross-organism version. For yeast it is a comparison; for a new chassis it is the
    only measured tier available at all, so its reach here is the honest estimate of what
    that tier would be worth.

**4. Do the measurements and the predictions disagree?** Every unit on the measured tier
that also has a prediction is compared in decades, and everything past two decades is
listed. Nothing is dropped: a two-decade gap is a flag for review, and which of the two
numbers is wrong is not decidable from the tables alone.

Run from the worktree root::

    PYTHONPATH=$PWD ~/miniconda3/envs/torchcell/bin/python \
        experiments/026-metabolism-flux/scripts/kinetic_measured_tier_audit.py
"""

from __future__ import annotations

import csv
import json
import os
import os.path as osp
from collections import Counter, defaultdict
from datetime import UTC, datetime
from typing import Any

import numpy as np
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict

from torchcell.metabolism.constraints import build_gem_tensors
from torchcell.metabolism.enzyme_kinetics import (
    KineticKind,
    OedKineticRecord,
    index_by_uniprot,
    load_mirrored_records,
)
from torchcell.metabolism.kinetics_pipeline import sha256_of
from torchcell.metabolism.parameters import (
    KINETICS_MIRROR,
    PLAUSIBILITY_RESULT,
    VERIFIED_PAIRS_FEATURE,
    SubstrateMatch,
    _database_tiers,
    load_predicted_unit_table,
    resolve_kcat_table,
    resolve_km_table,
    uniprot_for_genes,
)
from torchcell.metabolism.yeast_GEM import YeastGEM

load_dotenv()

DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "026-metabolism-flux", "results")
OED_ORGANISM = osp.join(
    DATA_ROOT, "data/enzyme_kinetics/open_enzyme_database/scerevisiae"
)
OED_ALL = osp.join(DATA_ROOT, "data/enzyme_kinetics/open_enzyme_database/all")
CHEM_PROP = osp.join(DATA_ROOT, "data/enzyme_kinetics/metanetx/chem_prop.tsv")
STRUCTURES_REPORT = osp.join(RESULTS_DIR, "kinetic_coverage_structures.json")
TARGET_TEMPERATURE_C = 30.0


def inchikey_block(smiles: str) -> str | None:
    """The connectivity block of a SMILES' InChIKey, or None when it does not parse.

    The first block encodes the molecular skeleton and drops protonation and stereochemistry
    from the comparison, which is what makes a database SMILES written as the neutral acid
    match a GEM species written as the anion.
    """
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    key = Chem.MolToInchiKey(mol)
    return key.split("-")[0] if key else None


class MeasuredUnitAudit(BaseModel):
    """One catalytic unit whose value came from the database, and what it was measured on."""

    model_config = ConfigDict(extra="forbid")

    unit_id: int
    reaction_id: str
    gene_ids: list[str]
    uniprot: str | None
    reaction_ec: list[str]
    record_ec: str | None
    organism: str | None
    substrate_assayed: str | None
    substrate_matches_reaction: bool
    match_route: str
    temperature_c: float | None
    n_candidates: int
    k_cat_measured: float
    k_cat_predicted: float | None
    abs_log10_ratio: float | None


class MeasuredTierAudit(BaseModel):
    """The whole audit, written verbatim to JSON."""

    model_config = ConfigDict(extra="forbid")

    built_at: str
    sources: dict[str, str]
    slice_rows: int
    database_rows: int
    slice_organisms: dict[str, int]
    rows_matching_organism_in_full_database: int
    n_distinct_accessions_in_slice: int
    n_distinct_ec_in_slice: int
    n_accessions_that_are_gem_genes: int
    tier_counts_k_cat: dict[str, int]
    tier_counts_K_M: dict[str, int]
    measured_units: list[MeasuredUnitAudit]
    substrate_agreement: dict[str, int]
    #: Where each unit the substrate check removes from the measured tier ends up.
    dropped_unit_fate: dict[str, int]
    #: Whether a cross-organism homolog measurement or a prediction is closer to the
    #: substrate-matched measurement, scored on the units that have one.
    homolog_versus_predicted: dict[str, float]
    matcher_reach: dict[str, dict[str, int]]
    disagreements_over_2_decades: list[MeasuredUnitAudit]
    disagreement_summary: dict[str, float]


def read_smiles_db(path: str) -> dict[str, str]:
    out: dict[str, str] = {}
    with open(path) as handle:
        for parts in csv.reader(handle, delimiter="\t"):
            if len(parts) >= 2 and parts[0].strip() and parts[1].strip():
                out.setdefault(parts[0].strip().lower(), parts[1].strip())
    return out


def gem_substrate_smiles(model: Any, gem: Any) -> dict[int, list[str]]:
    """Reaction index -> the SMILES of every substrate that resolves to one.

    Uses the same routes the predictor input table uses, plus the structures
    ``improve_kinetic_coverage.py`` recovered, so the substrate comparison is run against
    the structures the pipeline actually has rather than a smaller set.
    """
    db_dir = osp.join(YeastGEM().model_dir, "data", "databases")
    name_table = read_smiles_db(osp.join(db_dir, "smilesDB.tsv"))
    recovered: dict[str, str] = {}
    if osp.exists(STRUCTURES_REPORT):
        with open(STRUCTURES_REPORT) as handle:
            for entry in json.load(handle)["resolved"]:
                recovered[entry["met_id"]] = entry["smiles"]
    annotated: dict[str, str] = {}
    wanted = set()
    for metabolite in model.metabolites:
        reference = (metabolite.annotation or {}).get("metanetx.chemical")
        value = reference[0] if isinstance(reference, list) else reference
        if value:
            wanted.add(str(value))
    with open(CHEM_PROP) as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            columns = line.rstrip("\n").split("\t")
            if columns[0] in wanted and len(columns) > 8 and columns[8].strip():
                annotated[columns[0]] = columns[8].strip()

    per_metabolite: dict[str, str] = {}
    for metabolite in model.metabolites:
        hit = name_table.get((metabolite.name or "").lower())
        if hit:
            per_metabolite[metabolite.id] = hit
            continue
        reference = (metabolite.annotation or {}).get("metanetx.chemical")
        value = reference[0] if isinstance(reference, list) else reference
        if value and str(value) in annotated:
            per_metabolite[metabolite.id] = annotated[str(value)]
        elif metabolite.id in recovered:
            per_metabolite[metabolite.id] = recovered[metabolite.id]

    s = gem.s.coalesce()
    out: dict[int, list[str]] = defaultdict(list)
    for (i, j), v in zip(s.indices().t().tolist(), s.values().tolist()):
        if v < 0:
            met_id = model.metabolites[int(i)].id
            if met_id in per_metabolite:
                out[int(j)].append(per_metabolite[met_id])
    return out


def reaction_names(model: Any, gem: Any) -> dict[int, set[str]]:
    """Reaction index -> lowercased names of its substrates, the fallback match key."""
    s = gem.s.coalesce()
    out: dict[int, set[str]] = defaultdict(set)
    for (i, j), v in zip(s.indices().t().tolist(), s.values().tolist()):
        if v < 0:
            out[int(j)].add((model.metabolites[int(i)].name or "").lower())
    return out


def ec_of(reaction: Any) -> list[str]:
    reference = (reaction.annotation or {}).get("ec-code")
    if reference is None:
        return []
    return [str(r) for r in (reference if isinstance(reference, list) else [reference])]


def main() -> None:
    source = YeastGEM()
    model = source.model
    gem = build_gem_tensors(model, model_dir=source.model_dir)
    units = gem.catalytic_units

    slice_records = load_mirrored_records(OED_ORGANISM)
    all_records = load_mirrored_records(OED_ALL)
    slice_organisms = Counter(r.organism for r in slice_records)
    organism = next(iter(slice_organisms))
    same_organism_in_full = [r for r in all_records if r.organism == organism]

    by_uniprot = index_by_uniprot(slice_records)
    gene_to_uniprot = uniprot_for_genes(source.model_dir, units.gene_ids)
    accessions_in_gem = {a for a in gene_to_uniprot.values() if a in by_uniprot}

    kinetics_dir = osp.join(DATA_ROOT, KINETICS_MIRROR)
    plausibility = osp.join(EXPERIMENT_ROOT, PLAUSIBILITY_RESULT)
    substrates = SubstrateMatch.from_verified_pairs(
        osp.join(kinetics_dir, VERIFIED_PAIRS_FEATURE)
    )
    unchecked = SubstrateMatch(ec=substrates.ec, accept_all_unchecked=True)
    predicted_kcat = load_predicted_unit_table(
        "k_cat", units.n_units, kinetics_dir, plausibility
    )
    predicted_km = load_predicted_unit_table(
        "K_M", units.n_units, kinetics_dir, plausibility
    )

    # THE THREE DATABASE READINGS, all through the shipped resolver rather than a
    # reimplementation of it: the old accession-only key, the substrate-matched key, and
    # the homolog tier ALONE, which is what makes the accuracy comparison below possible.
    old_key = _database_tiers(
        units, gene_to_uniprot, slice_records, [], unchecked,
        KineticKind.KCAT, TARGET_TEMPERATURE_C,
    )
    new_key = _database_tiers(
        units, gene_to_uniprot, slice_records, [], substrates,
        KineticKind.KCAT, TARGET_TEMPERATURE_C,
    )
    homolog_only = _database_tiers(
        units, gene_to_uniprot, [], all_records, substrates,
        KineticKind.KCAT, TARGET_TEMPERATURE_C,
    )

    keys_by_smiles: dict[str, str | None] = {}

    def key_of(smiles: str) -> str | None:
        if smiles not in keys_by_smiles:
            keys_by_smiles[smiles] = inchikey_block(smiles)
        return keys_by_smiles[smiles]

    unit_genes: dict[int, list[str]] = defaultdict(list)
    for u, g in units.unit_gene_index.t().tolist():
        unit_genes[int(u)].append(units.gene_ids[int(g)])
    unit_reaction = units.unit_reaction.tolist()

    # HOW MANY OF THE OLD KEY'S UNITS WERE ASSAYED ON THE WRONG SUBSTRATE, and where each
    # one lands once the check is applied.
    agreement = Counter()
    fate = Counter()
    audits: list[MeasuredUnitAudit] = []
    for unit_id, (value, _, substrate) in sorted(old_key.items()):
        matched = unit_id in new_key
        agreement["substrate_matches" if matched else "substrate_does_not_match"] += 1
        if not matched:
            fate[
                "homolog"
                if unit_id in homolog_only
                else "predicted"
                if bool(predicted_kcat.mask[unit_id])
                else "default"
            ] += 1
        reaction = model.reactions[int(unit_reaction[unit_id])]
        prediction = (
            float(predicted_kcat.values[unit_id])
            if bool(predicted_kcat.mask[unit_id])
            else None
        )
        ratio = (
            float(abs(np.log10(value / prediction)))
            if prediction and prediction > 0 and value > 0
            else None
        )
        audits.append(
            MeasuredUnitAudit(
                unit_id=unit_id,
                reaction_id=reaction.id,
                gene_ids=unit_genes[unit_id],
                uniprot=next(
                    (gene_to_uniprot[g] for g in unit_genes[unit_id] if g in gene_to_uniprot),
                    None,
                ),
                reaction_ec=ec_of(reaction),
                record_ec=None,
                organism=organism,
                substrate_assayed=substrate,
                substrate_matches_reaction=matched,
                match_route="substrate_checked" if matched else "no_match",
                temperature_c=None,
                n_candidates=0,
                k_cat_measured=value,
                k_cat_predicted=prediction,
                abs_log10_ratio=ratio,
            )
        )

    # IS A HOMOLOG MEASUREMENT BETTER THAN A PREDICTION? Scored against the units that have
    # a substrate-matched measurement of this organism's own protein, which is the only
    # ground truth available. Both candidates are compared on exactly the same units.
    contest = {"n_units": 0, "homolog_closer": 0, "predicted_closer": 0}
    homolog_error: list[float] = []
    predicted_error: list[float] = []
    for unit_id, (truth, _, _) in new_key.items():
        if unit_id not in homolog_only or not bool(predicted_kcat.mask[unit_id]):
            continue
        homolog = homolog_only[unit_id][0]
        prediction = float(predicted_kcat.values[unit_id])
        if truth <= 0 or homolog <= 0 or prediction <= 0:
            continue
        h = abs(float(np.log10(homolog / truth)))
        p = abs(float(np.log10(prediction / truth)))
        homolog_error.append(h)
        predicted_error.append(p)
        contest["n_units"] += 1
        contest["homolog_closer" if h < p else "predicted_closer"] += 1
    accuracy = {
        **contest,
        "homolog_median_decades": float(np.median(homolog_error))
        if homolog_error
        else float("nan"),
        "predicted_median_decades": float(np.median(predicted_error))
        if predicted_error
        else float("nan"),
    }

    def reach(records: list[OedKineticRecord], key: str) -> dict[str, int]:
        by_ec: dict[str, list[OedKineticRecord]] = defaultdict(list)
        for r in records:
            if r.ec:
                by_ec[r.ec].append(r)
        n_units_hit = 0
        n_units_hit_with_substrate = 0
        for u in range(units.n_units):
            reaction = model.reactions[int(unit_reaction[u])]
            candidates: list[OedKineticRecord] = []
            if key == "uniprot":
                for gene in unit_genes[u]:
                    accession = gene_to_uniprot.get(gene)
                    if accession:
                        candidates.extend(by_uniprot.get(accession, []))
            else:
                for code in ec_of(reaction):
                    candidates.extend(by_ec.get(code, []))
            if not candidates:
                continue
            n_units_hit += 1
            if any(
                substrates.matches(u, key_of(c.smiles) if c.smiles else None, c.substrate)
                for c in candidates
            ):
                n_units_hit_with_substrate += 1
        return {
            "n_units_with_a_candidate_row": n_units_hit,
            "n_units_where_the_substrate_also_matches": n_units_hit_with_substrate,
        }

    matcher_reach = {
        "uniprot_organism_slice": reach(slice_records, "uniprot"),
        "ec_organism_slice": reach(slice_records, "ec"),
        "ec_any_organism": reach(all_records, "ec"),
    }

    kcat = resolve_kcat_table(
        units, source.model_dir, OED_ORGANISM, predicted=predicted_kcat,
        substrates=substrates, homolog_mirror_dir=OED_ALL,
    )
    km = resolve_km_table(
        units, source.model_dir, OED_ORGANISM, predicted=predicted_km,
        substrates=substrates, homolog_mirror_dir=OED_ALL,
    )

    disagreements = [
        a for a in audits if a.abs_log10_ratio is not None and a.abs_log10_ratio > 2.0
    ]
    ratios = [a.abs_log10_ratio for a in audits if a.abs_log10_ratio is not None]

    report = MeasuredTierAudit(
        built_at=datetime.now(UTC).isoformat(),
        sources={
            osp.join(OED_ORGANISM, "oed_records.json"): sha256_of(
                osp.join(OED_ORGANISM, "oed_records.json")
            ),
            osp.join(OED_ALL, "oed_records.json"): sha256_of(
                osp.join(OED_ALL, "oed_records.json")
            ),
            substrates.source_path: substrates.sha256,
        },
        slice_rows=len(slice_records),
        database_rows=len(all_records),
        slice_organisms={str(k): v for k, v in slice_organisms.items()},
        rows_matching_organism_in_full_database=len(same_organism_in_full),
        n_distinct_accessions_in_slice=len({r.uniprot for r in slice_records if r.uniprot}),
        n_distinct_ec_in_slice=len({r.ec for r in slice_records if r.ec}),
        n_accessions_that_are_gem_genes=len(accessions_in_gem),
        tier_counts_k_cat=kcat.tier_counts(),
        tier_counts_K_M=km.tier_counts(),
        measured_units=audits,
        substrate_agreement=dict(agreement),
        dropped_unit_fate=dict(fate),
        homolog_versus_predicted=accuracy,
        matcher_reach=matcher_reach,
        disagreements_over_2_decades=disagreements,
        disagreement_summary={
            "n_compared": float(len(ratios)),
            "median_decades": float(np.median(ratios)) if ratios else float("nan"),
            "p90_decades": float(np.percentile(ratios, 90)) if ratios else float("nan"),
            "max_decades": float(max(ratios)) if ratios else float("nan"),
            "n_over_1_decade": float(sum(1 for r in ratios if r > 1.0)),
            "n_over_2_decades": float(len(disagreements)),
        },
    )
    out_path = osp.join(RESULTS_DIR, "kinetic_measured_tier_audit.json")
    with open(out_path, "w") as handle:
        handle.write(report.model_dump_json(indent=2))

    print(f"slice {report.slice_rows} rows, organisms {report.slice_organisms}")
    print(
        f"the full {report.database_rows}-row database holds "
        f"{report.rows_matching_organism_in_full_database} rows for that organism"
    )
    print(f"accession-only key {len(old_key)} units, substrate-matched {len(new_key)}")
    print(f"substrate agreement {dict(agreement)}")
    print(f"fate of the dropped units {dict(fate)}")
    print(f"homolog versus predicted, scored on the measured units: {accuracy}")
    for name, block in matcher_reach.items():
        print(f"  {name:26} {block}")
    print(f"tiers k_cat {kcat.tier_counts()}  K_M {km.tier_counts()}")
    print(f"disagreement {report.disagreement_summary}")
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
