# experiments/026-metabolism-flux/scripts/kinetic_coverage_gaps.py
# [[experiments.026-metabolism-flux.scripts.kinetic_coverage_gaps]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/026-metabolism-flux/scripts/kinetic_coverage_gaps.py

r"""Why each catalytic unit is NOT on the measured or the fully predicted tier.

WHAT THIS ANSWERS THAT THE COVERAGE COUNTS DO NOT
-------------------------------------------------
``parameter_coverage.json`` says 3,552 of 3,728 catalytic units carry a prediction and
``resolve_kcat_table`` says 148 measured / 3,410 predicted / 170 default. Neither says
*why* a unit is on the default tier, and without that the gap is not actionable: a unit
blocked because its reaction has no substrate at all can never be predicted by a
``(sequence, substrate SMILES)`` model, while one blocked because yeast-GEM spells its
substrate as ``phosphatidylcholine (1-16:0, 2-16:1)`` is blocked by a naming convention
and is fixable.

So this script re-derives the tiers through the SAME functions the flux layer calls
(:func:`~torchcell.metabolism.parameters.resolve_kcat_table`,
:func:`~torchcell.metabolism.parameters.resolve_km_table`,
:func:`~torchcell.metabolism.parameters.load_predicted_unit_table`) and attaches one
:class:`Blocker` per unit that is neither measured nor predicted by every admitted
predictor. The tier numbers are therefore the flux layer's own, not a re-implementation
that could drift from it.

THE BLOCKER LADDER, IN THE ORDER IT IS APPLIED
----------------------------------------------
The order matters because a unit can carry several defects and only the first one that
would have to be fixed is worth reporting:

1. ``no_uniprot_mapping`` -- a member gene has no UniProt accession in the SwissProt table
   yeast-GEM ships, so there is no key into either a kinetics database or a predictor.
2. ``sequence_missing`` -- the accession exists but the row carries no protein sequence.
3. ``complex_rule_unhandled`` -- SOME member genes resolve and others do not. The per-unit
   builder requires every subunit, because the unit's turnover is the min over members and
   a missing member makes that min undefined rather than smaller.
4. ``reaction_has_no_substrate`` -- the unit's reaction consumes nothing (exchange,
   demand, sink or a pseudo-reaction). No substrate SMILES exists to be found.
5. ``substrate_structure_missing`` -- the reaction has substrates and none of them resolves
   to a SMILES through either the shipped ``smilesDB.tsv`` name table or the MetaNetX
   ``chem_prop.tsv`` id table.
6. ``predictor_failed`` -- inputs were complete and at least one admitted predictor still
   returned nothing for the unit; the predictor's own recorded failure reason is attached.
7. ``other`` -- inputs complete, no predictor failure recorded, still no value. Nothing is
   expected here and a nonzero count is a bug in this script or in the pipeline.

TWO POPULATIONS, DELIBERATELY SEPARATE
--------------------------------------
* **default-tier units** -- no measured value and no prediction from ANY admitted
  predictor. These are the units whose parameter is an organism constant.
* **units missing an all-selected prediction** -- at least one admitted predictor is silent,
  so the per-unit median is taken over fewer predictors than the selection admits. A unit
  can be in the second set and not the first, and the two have different fixes.

Run from the worktree root::

    PYTHONPATH=$PWD ~/miniconda3/envs/torchcell/bin/python \
        experiments/026-metabolism-flux/scripts/kinetic_coverage_gaps.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import os.path as osp
from collections import Counter, defaultdict
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field

from torchcell.metabolism.constraints import build_gem_tensors
from torchcell.metabolism.parameters import (
    KINETICS_MIRROR,
    PLAUSIBILITY_RESULT,
    VERIFIED_PAIRS_FEATURE,
    ParameterProvenance,
    PredictorSelection,
    SubstrateMatch,
    load_predicted_unit_table,
    load_swissprot,
    resolve_kcat_table,
    resolve_km_table,
)
from torchcell.metabolism.yeast_GEM import YeastGEM

load_dotenv()

DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "026-metabolism-flux", "results")
OED_MIRROR = osp.join(
    DATA_ROOT, "data/enzyme_kinetics/open_enzyme_database/scerevisiae"
)
METANETX_CHEM_PROP = osp.join(
    DATA_ROOT, "data/enzyme_kinetics/metanetx/chem_prop.tsv"
)

#: Extra SMILES written by ``lipid_structure_writer.py`` for the acyl-chain-specific lipid
#: names yeast-GEM enumerates. Read here only to REPORT how much of the substrate gap it
#: could close; it is applied by ``improve_kinetic_coverage.py``, not by this audit.
LIPID_STRUCTURES = osp.join(RESULTS_DIR, "lipid_structures.json")

#: Structures ``improve_kinetic_coverage.py --stage resolve`` recovered. When it exists,
#: those metabolites count as priced here too, so a rerun of this audit after the fix
#: reports the residual blockers rather than re-reporting the ones that were closed.
RESOLVED_STRUCTURES = osp.join(RESULTS_DIR, "kinetic_coverage_structures.json")

#: The composition-verified pair table. Once it exists it is AUTHORITATIVE for which
#: substrates have a structure: a SMILES that failed the composition check is not a
#: structure for that metabolite, whichever table it came from.
VERIFIED_PAIRS = osp.join(RESULTS_DIR, "kinetic_verified_pairs.parquet")


def residual_substrate_reason(formula: str, name: str, n_candidates: int | None) -> str:
    """Why one substrate has no small-molecule structure, at the resolution a fix needs.

    Four situations with four different fixes. The last two are separated by evidence
    rather than by a name heuristic: whether the cross-reference route offered any
    candidate structure at all. **No candidate** means the catalogs do not carry the
    compound, and writing its structure closes it. **Candidates that all fail the
    composition test** means the catalogs do carry it and the GEM's formula is a partial
    species, typically a protein-bound cofactor written as the cofactor alone, so no lookup
    can close it and someone has to decide what the GEM means.
    """
    if not formula:
        return "macromolecular species with no formula in the GEM"
    if "R" in formula or "*" in formula:
        return "R-group placeholder, the GEM defines the species only up to a substituent"
    if name.strip().startswith("[") or "-[" in name:
        return "protein-bound species named as an enzyme state, not a compound"
    if n_candidates:
        return (
            "candidate structures exist but none matches the GEM's composition, so the "
            "GEM's formula is a partial or protein-bound species"
        )
    return "no candidate structure in any mirrored catalog for this compound"


class Blocker(StrEnum):
    """Why one catalytic unit has no value on the tier being audited."""

    NO_UNIPROT_MAPPING = "no_uniprot_mapping"
    SEQUENCE_MISSING = "sequence_missing"
    COMPLEX_RULE_UNHANDLED = "complex_rule_unhandled"
    REACTION_HAS_NO_SUBSTRATE = "reaction_has_no_substrate"
    SUBSTRATE_STRUCTURE_MISSING = "substrate_structure_missing"
    PREDICTOR_FAILED = "predictor_failed"
    OTHER = "other"


class BlockedUnit(BaseModel):
    """One catalytic unit that is on neither the measured nor the fully predicted tier."""

    model_config = ConfigDict(extra="forbid")

    unit_id: int
    reaction_id: str
    reaction_name: str
    gene_ids: list[str]
    n_genes: int
    ec_numbers: list[str]
    blocker: Blocker
    detail: str
    #: Substrate metabolites of the unit's reaction that carry no SMILES, as
    #: ``(met_id, name, formula)``. Empty unless the blocker is a structure one.
    unresolved_substrates: list[tuple[str, str, str]] = Field(default_factory=list)
    #: Admitted predictors that produced nothing for this unit.
    missing_predictors: list[str] = Field(default_factory=list)


class ParameterGapReport(BaseModel):
    """The gap accounting for one parameter, both populations."""

    model_config = ConfigDict(extra="forbid")

    parameter: str
    selected_predictors: list[str]
    selection_rule: str
    selection_baseline_fraction: float
    tier_counts: dict[str, int]
    n_units: int
    n_units_any_selected_prediction: int
    n_units_all_selected_predictions: int
    n_units_partial_selected_predictions: int
    n_units_no_selected_prediction: int
    default_tier_blockers: dict[str, int]
    default_tier_units: list[BlockedUnit]
    missing_all_selected_blockers: dict[str, int]
    missing_all_selected_units: list[BlockedUnit]
    predictor_row_failures: dict[str, dict[str, int]]


class CoverageGapReport(BaseModel):
    """The whole audit, one object, written verbatim to JSON."""

    model_config = ConfigDict(extra="forbid")

    built_at: str
    n_catalytic_units: int
    n_reactions: int
    n_metabolites: int
    n_genes: int
    sources: dict[str, str] = Field(
        description="Every file the audit read, path -> sha256."
    )
    substrate_structure_routes: dict[str, int]
    lipid_writer_reach: dict[str, int]
    parameters: dict[str, ParameterGapReport]


def sha256_of(path: str, chunk: int = 1 << 20) -> str:
    """sha256 of a file the audit read, so the report pins its own inputs."""
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(chunk):
            h.update(block)
    return h.hexdigest()


def read_smiles_db(path: str) -> dict[str, str]:
    """Metabolite name (lowercased) -> SMILES, from the yeast-GEM shipped name table."""
    out: dict[str, str] = {}
    with open(path) as handle:
        for parts in csv.reader(handle, delimiter="\t"):
            if len(parts) < 2:
                continue
            name, smiles = parts[0].strip(), parts[1].strip()
            if name and smiles:
                out.setdefault(name.lower(), smiles)
    return out


def read_metanetx_smiles(path: str) -> dict[str, str]:
    """MNXM id -> SMILES from the mirrored MetaNetX ``chem_prop.tsv``."""
    out: dict[str, str] = {}
    with open(path) as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) > 8 and cols[0] and cols[8].strip():
                out[cols[0]] = cols[8].strip()
    return out


def metanetx_id(met: Any) -> str | None:
    """The metabolite's MetaNetX id; cobra holds an annotation as a string or a list."""
    ref = (met.annotation or {}).get("metanetx.chemical")
    if isinstance(ref, list):
        return ref[0] if ref else None
    return ref


def ec_numbers(reaction: Any) -> list[str]:
    """EC numbers annotated on a reaction, normalized to a list of strings."""
    ref = (reaction.annotation or {}).get("ec-code")
    if ref is None:
        return []
    return [str(r) for r in (ref if isinstance(ref, list) else [ref])]


def per_unit_predictor_mask(
    parquet_path: str, column: str, n_units: int
) -> np.ndarray:
    """Bool per unit: did this one predictor emit a usable value for the unit?

    The positivity and finiteness filter is the same one
    :func:`~torchcell.metabolism.parameters._per_unit_from_predictor` applies, so a unit
    counted here as covered is a unit the median actually uses.
    """
    frame = pd.read_parquet(parquet_path, columns=["unit_id", column])
    frame = frame[np.isfinite(frame[column]) & (frame[column] > 0)]
    mask = np.zeros(n_units, dtype=bool)
    ids = frame["unit_id"].to_numpy().astype(int)
    if ids.size:
        mask[ids] = True
    return mask


def main() -> None:
    source = YeastGEM()
    model = source.model
    gem = build_gem_tensors(model, model_dir=source.model_dir)
    units = gem.catalytic_units
    n_units = units.n_units

    db_dir = osp.join(source.model_dir, "data", "databases")
    swissprot_path = osp.join(db_dir, "swissprot.tsv")
    smiles_db_path = osp.join(db_dir, "smilesDB.tsv")
    swissprot = load_swissprot(source.model_dir)
    smiles_db = read_smiles_db(smiles_db_path)
    mnx_smiles = read_metanetx_smiles(METANETX_CHEM_PROP)

    # --- substrate structures, by route ------------------------------------------------
    recovered: dict[str, str] = {}
    candidates_seen: dict[str, int] = {}
    if osp.exists(RESOLVED_STRUCTURES):
        with open(RESOLVED_STRUCTURES) as handle:
            report_json = json.load(handle)
        for entry in report_json["resolved"]:
            recovered[entry["met_id"]] = entry["route"]
        for entry in report_json["unresolved"]:
            candidates_seen[entry["met_id"]] = int(entry["n_candidates"])

    verified_substrates: set[str] = set()
    if osp.exists(VERIFIED_PAIRS):
        verified_substrates = set(
            pd.read_parquet(VERIFIED_PAIRS, columns=["substrate_met_id"])[
                "substrate_met_id"
            ].unique()
        )

    met_smiles: dict[int, str] = {}
    route_counts = Counter()
    for k, met in enumerate(model.metabolites):
        # The verified table decides first, and it can REMOVE a structure the older routes
        # accepted: a name match whose composition disagrees with the GEM's formula is a
        # structure for a different molecule.
        if verified_substrates:
            if met.id in verified_substrates:
                met_smiles[k] = "verified"
                route_counts["verified"] += 1
            else:
                route_counts["none"] += 1
            continue
        hit = smiles_db.get((met.name or "").lower())
        if hit:
            met_smiles[k] = "smilesDB"
            route_counts["smilesDB"] += 1
            continue
        mnx = metanetx_id(met)
        if mnx and mnx in mnx_smiles:
            met_smiles[k] = "metanetx"
            route_counts["metanetx"] += 1
        elif met.id in recovered:
            met_smiles[k] = recovered[met.id]
            route_counts[recovered[met.id]] += 1
        else:
            route_counts["none"] += 1

    s = gem.s.coalesce()
    rxn_substrates: dict[int, list[int]] = defaultdict(list)
    for (i, j), v in zip(s.indices().t().tolist(), s.values().tolist()):
        if v < 0:
            rxn_substrates[int(j)].append(int(i))

    unit_genes: dict[int, list[str]] = defaultdict(list)
    for u, g in units.unit_gene_index.t().tolist():
        unit_genes[int(u)].append(units.gene_ids[int(g)])
    unit_reaction = units.unit_reaction.tolist()

    # --- what the lipid writer could add ----------------------------------------------
    lipid_met_ids: set[str] = set()
    if osp.exists(LIPID_STRUCTURES):
        with open(LIPID_STRUCTURES) as handle:
            for entry in json.load(handle)["structures"]:
                if entry.get("formula_match"):
                    lipid_met_ids.update(entry["met_ids"])

    # --- tiers, through the flux layer's own resolution path ---------------------------
    kinetics_dir = osp.join(DATA_ROOT, KINETICS_MIRROR)
    plausibility = osp.join(EXPERIMENT_ROOT, PLAUSIBILITY_RESULT)
    predicted = {
        "k_cat": load_predicted_unit_table("k_cat", n_units, kinetics_dir, plausibility),
        "K_M": load_predicted_unit_table("K_M", n_units, kinetics_dir, plausibility),
    }
    substrates = SubstrateMatch.from_verified_pairs(
        osp.join(kinetics_dir, VERIFIED_PAIRS_FEATURE)
    )
    homolog_mirror = osp.join(
        DATA_ROOT, "data/enzyme_kinetics/open_enzyme_database/all"
    )
    tables = {
        "k_cat": resolve_kcat_table(
            units,
            source.model_dir,
            OED_MIRROR,
            predicted=predicted["k_cat"],
            substrates=substrates,
            homolog_mirror_dir=homolog_mirror,
        ),
        "K_M": resolve_km_table(
            units,
            source.model_dir,
            OED_MIRROR,
            predicted=predicted["K_M"],
            substrates=substrates,
            homolog_mirror_dir=homolog_mirror,
        ),
    }

    sources: dict[str, str] = {
        swissprot_path: sha256_of(swissprot_path),
        smiles_db_path: sha256_of(smiles_db_path),
        METANETX_CHEM_PROP: sha256_of(METANETX_CHEM_PROP),
        osp.join(OED_MIRROR, "oed_records.json"): sha256_of(
            osp.join(OED_MIRROR, "oed_records.json")
        ),
        plausibility: sha256_of(plausibility),
    }
    if osp.exists(LIPID_STRUCTURES):
        sources[LIPID_STRUCTURES] = sha256_of(LIPID_STRUCTURES)

    reports: dict[str, ParameterGapReport] = {}
    for parameter, table in tables.items():
        selection: PredictorSelection = predicted[parameter].selection
        sources.update(
            {
                predicted[parameter].source_paths[name]: digest
                for name, digest in predicted[parameter].sha256.items()
            }
        )

        per_predictor = {
            name: per_unit_predictor_mask(path, parameter, n_units)
            for name, path in predicted[parameter].source_paths.items()
        }
        stacked = np.vstack([per_predictor[n] for n in selection.selected])
        n_hit = stacked.sum(axis=0)
        all_selected = n_hit == len(selection.selected)
        any_selected = n_hit > 0

        row_failures: dict[str, dict[str, int]] = {}
        for name in selection.selected:
            manifest_path = osp.join(
                kinetics_dir, name, predicted[parameter].version, f"{parameter}_manifest.json"
            )
            with open(manifest_path) as handle:
                manifest = json.load(handle)
            row_failures[name] = {
                "pairs_the_predictor_left_unpriced": int(
                    manifest.get("n_pairs_unpriced", 0)
                )
            }
            sources[manifest_path] = sha256_of(manifest_path)

        def classify(u: int) -> BlockedUnit:
            """The first defect on the ladder that would have to be fixed for unit ``u``."""
            j = int(unit_reaction[u])
            reaction = model.reactions[j]
            genes = unit_genes[u]
            resolved = [g for g in genes if g in swissprot]
            no_accession = [
                g for g in resolved if not (swissprot[g].get("uniprot") or "").strip()
            ]
            no_sequence = [
                g for g in resolved if not (swissprot[g].get("sequence") or "").strip()
            ]
            substrates = rxn_substrates.get(j, [])
            without_smiles = [i for i in substrates if i not in met_smiles]
            missing_predictors = [
                name for name in selection.selected if not per_predictor[name][u]
            ]

            if len(resolved) < len(genes) and not resolved:
                blocker, detail = (
                    Blocker.NO_UNIPROT_MAPPING,
                    "no member gene is in the SwissProt table yeast-GEM ships",
                )
            elif len(resolved) < len(genes):
                blocker, detail = (
                    Blocker.COMPLEX_RULE_UNHANDLED,
                    f"{len(genes) - len(resolved)} of {len(genes)} subunits unresolved; "
                    "the per-unit builder requires every subunit",
                )
            elif no_accession:
                blocker, detail = (
                    Blocker.NO_UNIPROT_MAPPING,
                    f"SwissProt rows for {', '.join(no_accession)} carry no accession",
                )
            elif no_sequence:
                blocker, detail = (
                    Blocker.SEQUENCE_MISSING,
                    f"SwissProt rows for {', '.join(no_sequence)} carry no sequence",
                )
            elif not substrates:
                blocker, detail = (
                    Blocker.REACTION_HAS_NO_SUBSTRATE,
                    f"{reaction.id} consumes nothing "
                    f"({'exchange' if reaction.boundary else 'pseudo-reaction'}); a "
                    "(sequence, substrate SMILES) predictor has no input",
                )
            elif len(without_smiles) == len(substrates):
                in_lipid = [
                    model.metabolites[i].id
                    for i in without_smiles
                    if model.metabolites[i].id in lipid_met_ids
                ]
                blocker = Blocker.SUBSTRATE_STRUCTURE_MISSING
                reasons = sorted(
                    {
                        residual_substrate_reason(
                            model.metabolites[i].formula or "",
                            model.metabolites[i].name or "",
                            candidates_seen.get(model.metabolites[i].id),
                        )
                        for i in without_smiles
                    }
                )
                detail = (
                    f"none of {len(substrates)} substrates resolves to a SMILES through "
                    f"any mirrored route; {len(in_lipid)} are written by "
                    # Separated by a pipe, not a semicolon: a reason may itself contain a
                    # semicolon, and a consumer splitting on one gets half-sentences.
                    f"lipid_structure_writer.py; residual reasons: {' | '.join(reasons)}"
                )
            elif missing_predictors:
                reasons = "; ".join(
                    f"{name}: "
                    + (
                        ", ".join(f"{k} x{v}" for k, v in row_failures[name].items())
                        or "no recorded failure reason"
                    )
                    for name in missing_predictors
                )
                blocker, detail = (Blocker.PREDICTOR_FAILED, reasons)
            else:
                blocker, detail = (
                    Blocker.OTHER,
                    "inputs complete and every admitted predictor emitted a value; "
                    "reaching this branch is a bug",
                )
            return BlockedUnit(
                unit_id=u,
                reaction_id=reaction.id,
                reaction_name=reaction.name or "",
                gene_ids=genes,
                n_genes=len(genes),
                ec_numbers=ec_numbers(reaction),
                blocker=blocker,
                detail=detail,
                unresolved_substrates=[
                    (
                        model.metabolites[i].id,
                        model.metabolites[i].name or "",
                        model.metabolites[i].formula or "",
                    )
                    for i in without_smiles
                ],
                missing_predictors=missing_predictors,
            )

        default_units = [
            classify(u)
            for u, p in enumerate(table.provenance)
            if p is ParameterProvenance.ORGANISM_DEFAULT
        ]
        missing_all = [classify(u) for u in range(n_units) if not all_selected[u]]

        reports[parameter] = ParameterGapReport(
            parameter=parameter,
            selected_predictors=selection.selected,
            selection_rule=selection.statistic,
            selection_baseline_fraction=selection.baseline_fraction,
            tier_counts=table.tier_counts(),
            n_units=n_units,
            n_units_any_selected_prediction=int(any_selected.sum()),
            n_units_all_selected_predictions=int(all_selected.sum()),
            n_units_partial_selected_predictions=int(
                (any_selected & ~all_selected).sum()
            ),
            n_units_no_selected_prediction=int((~any_selected).sum()),
            default_tier_blockers=dict(
                Counter(str(b.blocker) for b in default_units).most_common()
            ),
            default_tier_units=default_units,
            missing_all_selected_blockers=dict(
                Counter(str(b.blocker) for b in missing_all).most_common()
            ),
            missing_all_selected_units=missing_all,
            predictor_row_failures=row_failures,
        )

    unpriced_substrate_ids = {
        met_id
        for report in reports.values()
        for unit in report.default_tier_units
        for met_id, _, _ in unit.unresolved_substrates
    }
    report = CoverageGapReport(
        built_at=datetime.now(UTC).isoformat(),
        n_catalytic_units=n_units,
        n_reactions=len(model.reactions),
        n_metabolites=len(model.metabolites),
        n_genes=len(units.gene_ids),
        sources=sources,
        substrate_structure_routes=dict(route_counts),
        lipid_writer_reach={
            "n_lipid_structures_formula_verified": len(lipid_met_ids),
            "n_substrates_of_default_tier_units_without_smiles": len(
                unpriced_substrate_ids
            ),
            "n_of_those_the_lipid_writer_covers": len(
                unpriced_substrate_ids & lipid_met_ids
            ),
        },
        parameters=reports,
    )

    out_path = osp.join(RESULTS_DIR, "kinetic_coverage_gaps.json")
    with open(out_path, "w") as handle:
        handle.write(report.model_dump_json(indent=2))

    for parameter, block in reports.items():
        print(f"\n=== {parameter} ===")
        print(f"tiers                        {block.tier_counts}")
        print(f"selected                     {block.selected_predictors}")
        print(f"all selected predictions     {block.n_units_all_selected_predictions}")
        print(f"partial                      {block.n_units_partial_selected_predictions}")
        print(f"none                         {block.n_units_no_selected_prediction}")
        print("default-tier blockers:")
        for name, count in block.default_tier_blockers.items():
            print(f"  {name:34} {count}")
        print("missing-all-selected blockers:")
        for name, count in block.missing_all_selected_blockers.items():
            print(f"  {name:34} {count}")
    print(f"\nlipid writer reach {report.lipid_writer_reach}")
    print(f"\n-> {out_path}")


if __name__ == "__main__":
    main()
