# experiments/026-metabolism-flux/scripts/improve_kinetic_coverage.py
# [[experiments.026-metabolism-flux.scripts.improve_kinetic_coverage]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/026-metabolism-flux/scripts/improve_kinetic_coverage.py

r"""Close the fixable half of the kinetic-parameter gap, one blocker class at a time.

WHAT THE AUDIT LEFT
--------------------
``kinetic_coverage_gaps.py`` finds that every one of the 176 catalytic units without a
prediction is blocked by the same thing: **no substrate of the unit's reaction resolves to
a SMILES**. Not a missing sequence, not a missing UniProt accession, not a predictor
failure, and not a reaction without substrates. So there is exactly one fix worth building,
and this script builds it.

THREE ROUTES, IN THIS ORDER, EACH ONE FORMULA-CHECKED
------------------------------------------------------
1. **cross-reference** -- yeast-GEM annotates metabolites against an older MNXref release,
   so ``MNXM417`` for fructose 1,6-bisphosphate is not a row in MNXref 4.5 at all and the
   direct-id lookup the input builder does silently returns nothing. Going
   ``chebi/kegg/bigg/mnx id -> chem_xref.tsv -> current MNXM id -> chem_prop.tsv`` recovers
   the structure. This is the route that unblocks central metabolism: fructose-bisphosphate
   aldolase, the IMP cyclohydrolase step, D-arabinose transport.
2. **supplied structures** -- the acyl-chain-specific lipid species yeast-GEM enumerates
   combinatorially (``phosphatidylcholine (1-16:0, 2-16:1)``) are named, not catalogued, so
   no name table and no cross-reference reaches them. ``lipid_structure_writer.py`` builds
   them from the acyl-chain grammar and this script reads its formula-verified output.
3. Neither -- the substrate is a macromolecule or an R-group placeholder
   (``Arg-tRNA(Arg)``, ``3-hydroxybutanoyl-ACP``, ``generic protein``). There is no single
   small-molecule structure to find, so the unit stays on the default tier WITH its blocker
   recorded. Nothing is invented for it.

**Every candidate is accepted only if RDKit's formula for the SMILES equals the GEM's
formula for the metabolite, both neutralized.** See
:mod:`torchcell.metabolism.kinetics_pipeline` for why that test is the whole game.

ADDITIVE BY CONSTRUCTION
------------------------
A metabolite the existing pipeline already priced is never re-resolved, even where the new
routes would offer a different structure. Re-pricing would move the per-unit median of the
3,552 units that already have predictions, which would silently invalidate the predictor
selection study and the spread analysis that were computed on them. The counterfactual is
measured and reported instead of applied.

STAGES
------
``--stage resolve``
    Resolve the blocked units' substrates, write ``kinetic_coverage_structures.json`` and
    the new pair rows as ``kinetic_coverage_new_pairs.parquet``.
``--stage predict --predictor <name>``
    Run one admitted predictor over the new rows only and append to that predictor's
    parquet, with an append manifest recording exactly which rows were added.
``--stage report``
    Re-resolve both tables through the flux layer's own path and write the before/after.

Run from the worktree root::

    PYTHONPATH=$PWD ~/miniconda3/envs/torchcell/bin/python \
        experiments/026-metabolism-flux/scripts/improve_kinetic_coverage.py --stage resolve
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import os.path as osp
import sys
from collections import defaultdict
from datetime import UTC, datetime
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict

from torchcell.metabolism.constraints import build_gem_tensors
from torchcell.metabolism.kinetics_pipeline import (
    CrossReferenceStructureIndex,
    ResolvedStructure,
    SubstrateStructureResolver,
    read_supplied_structures,
    sha256_of,
    wanted_xrefs_for,
)
from torchcell.metabolism.parameters import (
    KINETICS_MIRROR,
    PLAUSIBILITY_RESULT,
    VERIFIED_PAIRS_FEATURE,
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
CHEM_PROP = osp.join(DATA_ROOT, "data/enzyme_kinetics/metanetx/chem_prop.tsv")
CHEM_XREF = osp.join(DATA_ROOT, "data/enzyme_kinetics/metanetx/chem_xref.tsv")
LIPID_STRUCTURES = osp.join(RESULTS_DIR, "lipid_structures.json")
#: The kinetics-side structure writer's output, for the substrates no catalog carries and
#: the lipid writer's grammar does not cover. Same schema, so the two merge.
KINETIC_STRUCTURES = osp.join(RESULTS_DIR, "kinetic_substrate_structures.json")
KINETICS_ROOT = osp.join(DATA_ROOT, KINETICS_MIRROR)
#: The protein sequences every predictor in the comparison was fed, dumped from the genome
#: object by ``embed_prot_t5.py --dump-sequences``. New rows must reuse these byte for byte
#: or a predictor sees a different protein for the same gene than it saw in the first run.
GENOME_SEQUENCES = osp.join(KINETICS_ROOT, "_features", "gem_protein_sequences.json")

NEW_PAIRS = osp.join(RESULTS_DIR, "kinetic_coverage_new_pairs.parquet")
STRUCTURES_REPORT = osp.join(RESULTS_DIR, "kinetic_coverage_structures.json")
CLOSURE_REPORT = osp.join(RESULTS_DIR, "kinetic_coverage_closure.json")

#: The composition-verified pair table, and the report that explains how it differs from
#: the unverified one. Written by ``--stage verify``.
VERIFIED_PAIRS = osp.join(RESULTS_DIR, "kinetic_verified_pairs.parquet")
VERIFICATION_REPORT = osp.join(RESULTS_DIR, "kinetic_structure_verification.json")

#: Subdirectory of each predictor's ``processed/`` holding the verified-structure version
#: of its table. The unverified version stays where it is, untouched, so before and after
#: are both on disk and the manifest names the sha256 of what each version supersedes.
VERIFIED_SUBDIR = "verified"

#: Every predictor with a table, per parameter. The four the within-class study does not
#: admit are here too: the selection rule can only be re-decided later on a full matrix,
#: and that needs every predictor to cover the same units.
ALL_PREDICTORS = {
    "k_cat": [
        "ezkinetics",
        "eitlem",
        "unikp",
        "dlkcat",
        "turnup",
        "deepenzyme",
        "catpred",
    ],
    "K_M": ["ezkinetics", "eitlem", "unikp", "boost_km", "catpred"],
}

#: The predictor whose table defines the reference pair set: the only one that carries
#: every pair, including the multi-component SMILES UniKP refuses.
REFERENCE_PREDICTOR = "eitlem"


class StructureResolutionReport(BaseModel):
    """What the new routes recovered, and what they could not."""

    model_config = ConfigDict(extra="forbid")

    built_at: str
    sources: dict[str, str]
    n_metabolites: int
    n_priced_by_existing_pipeline: int
    n_unpriced_by_existing_pipeline: int
    n_substrates_of_blocked_units: int
    n_blocked_units_before: int
    resolved: list[ResolvedStructure]
    unresolved: list[dict[str, str]]
    resolved_by_route: dict[str, int]
    n_blocked_units_recoverable: int
    n_new_pair_rows: int
    #: Metabolites outside the blocked units that the new routes COULD price. Measured and
    #: reported, deliberately not applied; see the module docstring.
    counterfactual_other_metabolites_resolvable: int
    counterfactual_other_metabolites_examined: int


def write_parquet_atomically(frame: pd.DataFrame, path: str) -> None:
    """Write a parquet the flux layer may be reading, without a torn intermediate state.

    Training cells re-import the resolver in spawned DataLoader workers, so a table can be
    read at any moment. ``to_parquet`` truncates in place, which gives a reader a window on
    a partial file; writing beside it and renaming makes the swap atomic on one filesystem.
    """
    temporary = f"{path}.tmp"
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def supplied_structures() -> dict[str, str]:
    """Every externally written structure, from both writers, keyed by metabolite id.

    Two writers with one schema. The lipid writer covers the acyl-chain-specific species a
    GEM enumerates combinatorially; ``kinetic_substrate_writer.py`` covers the named
    compounds no mirrored catalog indexes. Both formula-check their own output and the
    resolver checks it again, so a collision between them cannot admit a wrong structure,
    only pick one of two right ones.
    """
    merged = read_supplied_structures(LIPID_STRUCTURES)
    merged.update(read_supplied_structures(KINETIC_STRUCTURES))
    return merged


def read_smiles_db(path: str) -> dict[str, str]:
    """Metabolite name (lowercased) -> SMILES, from the table the GEM ships."""
    out: dict[str, str] = {}
    with open(path) as handle:
        for parts in csv.reader(handle, delimiter="\t"):
            if len(parts) < 2:
                continue
            name, smiles = parts[0].strip(), parts[1].strip()
            if name and smiles:
                out.setdefault(name.lower(), smiles)
    return out


def read_metanetx_direct(path: str, wanted: set[str]) -> dict[str, str]:
    """MNXM id -> SMILES for the ids a GEM annotates, the existing pipeline's second route."""
    out: dict[str, str] = {}
    with open(path) as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            columns = line.rstrip("\n").split("\t")
            if columns[0] in wanted and len(columns) > 8 and columns[8].strip():
                out[columns[0]] = columns[8].strip()
    return out


def metanetx_id(metabolite: Any) -> str | None:
    """The metabolite's MetaNetX id as the existing input builder reads it."""
    reference = (metabolite.annotation or {}).get("metanetx.chemical")
    if isinstance(reference, list):
        return reference[0] if reference else None
    return reference


def existing_pipeline_smiles(model: Any) -> dict[str, tuple[str, str]]:
    """Metabolite id -> (SMILES, route) exactly as ``kinetics_input_audit.py`` resolves it.

    Reproduced rather than imported so the "before" state is the state on disk, not a state
    that changes when the audit script changes.
    """
    db_dir = osp.join(YeastGEM().model_dir, "data", "databases")
    name_table = read_smiles_db(osp.join(db_dir, "smilesDB.tsv"))
    annotated = {
        mnx for m in model.metabolites if (mnx := metanetx_id(m)) is not None
    }
    direct = read_metanetx_direct(CHEM_PROP, annotated)
    out: dict[str, tuple[str, str]] = {}
    for metabolite in model.metabolites:
        hit = name_table.get((metabolite.name or "").lower())
        if hit:
            out[metabolite.id] = (hit, "smilesDB")
            continue
        mnx = metanetx_id(metabolite)
        if mnx and mnx in direct:
            out[metabolite.id] = (direct[mnx], "metanetx")
    return out


def stage_resolve() -> None:
    """Find structures for the blocked units' substrates and write the new pair rows."""
    source = YeastGEM()
    model = source.model
    gem = build_gem_tensors(model, model_dir=source.model_dir)
    units = gem.catalytic_units

    priced = existing_pipeline_smiles(model)
    print(f"existing pipeline prices {len(priced)} of {len(model.metabolites)} metabolites")

    s = gem.s.coalesce()
    rxn_substrates: dict[int, list[int]] = defaultdict(list)
    for (i, j), v in zip(s.indices().t().tolist(), s.values().tolist()):
        if v < 0:
            rxn_substrates[int(j)].append(int(i))
    unit_genes: dict[int, list[str]] = defaultdict(list)
    for u, g in units.unit_gene_index.t().tolist():
        unit_genes[int(u)].append(units.gene_ids[int(g)])
    unit_reaction = units.unit_reaction.tolist()

    blocked_units = [
        u
        for u in range(units.n_units)
        if rxn_substrates.get(int(unit_reaction[u]))
        and all(
            model.metabolites[i].id not in priced
            for i in rxn_substrates[int(unit_reaction[u])]
        )
    ]
    blocked_substrates = {
        model.metabolites[i].id
        for u in blocked_units
        for i in rxn_substrates[int(unit_reaction[u])]
    }
    print(f"{len(blocked_units)} blocked units, {len(blocked_substrates)} substrates")

    unpriced = [m for m in model.metabolites if m.id not in priced]
    index = CrossReferenceStructureIndex.from_metanetx(
        CHEM_PROP, CHEM_XREF, wanted_xrefs_for(unpriced)
    )
    print(
        f"cross-reference index: {len(index.ids_by_xref)} keys -> "
        f"{len(index.smiles_by_id)} structures"
    )
    resolver = SubstrateStructureResolver(
        index=index, supplied=read_supplied_structures(LIPID_STRUCTURES)
    )

    resolved: dict[str, ResolvedStructure] = {}
    unresolved: list[dict[str, str]] = []
    for met_id in sorted(blocked_substrates):
        metabolite = model.metabolites.get_by_id(met_id)
        hit = resolver.resolve(metabolite)
        if hit is not None:
            resolved[met_id] = hit
        else:
            unresolved.append(
                {
                    "met_id": met_id,
                    "name": metabolite.name or "",
                    "formula": metabolite.formula or "",
                    "n_candidates": str(len(resolver.candidates(metabolite))),
                }
            )
    print(f"resolved {len(resolved)} of {len(blocked_substrates)} blocked substrates")

    # --- the counterfactual, measured and not applied ---------------------------------
    others = [
        m
        for m in unpriced
        if m.id not in blocked_substrates
    ]
    counterfactual = sum(1 for m in others if resolver.resolve(m) is not None)
    print(f"counterfactual: {counterfactual} of {len(others)} other unpriced metabolites")

    # --- new pair rows ------------------------------------------------------------------
    with open(GENOME_SEQUENCES) as handle:
        sequences = {k.upper(): v for k, v in json.load(handle).items()}
    swissprot = load_swissprot(source.model_dir)

    rows: list[dict[str, Any]] = []
    recovered_units: set[int] = set()
    for u in blocked_units:
        j = int(unit_reaction[u])
        substrates = [
            resolved[model.metabolites[i].id]
            for i in rxn_substrates[j]
            if model.metabolites[i].id in resolved
        ]
        if not substrates:
            continue
        genes = unit_genes[u]
        if any(g.upper() not in sequences for g in genes):
            continue
        recovered_units.add(u)
        for gene in genes:
            for structure in substrates:
                metabolite = model.metabolites.get_by_id(structure.met_id)
                rows.append(
                    {
                        "unit_id": u,
                        "reaction_id": model.reactions[j].id,
                        "gene_id": gene,
                        "uniprot": (swissprot.get(gene) or {}).get("uniprot", ""),
                        "sequence_length": len(sequences[gene.upper()]),
                        "substrate_met_id": structure.met_id,
                        "substrate_name": metabolite.name or "",
                        "smiles": structure.smiles,
                        "smiles_source": str(structure.route),
                        "sequence": sequences[gene.upper()],
                    }
                )
    frame = pd.DataFrame(rows)
    frame.to_parquet(NEW_PAIRS, index=False)
    print(f"{len(frame)} new pair rows over {len(recovered_units)} units -> {NEW_PAIRS}")

    report = StructureResolutionReport(
        built_at=datetime.now(UTC).isoformat(),
        sources={
            **index.source_paths,
            **(
                {LIPID_STRUCTURES: sha256_of(LIPID_STRUCTURES)}
                if osp.exists(LIPID_STRUCTURES)
                else {}
            ),
            GENOME_SEQUENCES: sha256_of(GENOME_SEQUENCES),
        },
        n_metabolites=len(model.metabolites),
        n_priced_by_existing_pipeline=len(priced),
        n_unpriced_by_existing_pipeline=len(unpriced),
        n_substrates_of_blocked_units=len(blocked_substrates),
        n_blocked_units_before=len(blocked_units),
        resolved=list(resolved.values()),
        unresolved=unresolved,
        resolved_by_route=dict(
            pd.Series([str(r.route) for r in resolved.values()]).value_counts()
        ),
        n_blocked_units_recoverable=len(recovered_units),
        n_new_pair_rows=len(frame),
        counterfactual_other_metabolites_resolvable=counterfactual,
        counterfactual_other_metabolites_examined=len(others),
    )
    with open(STRUCTURES_REPORT, "w") as handle:
        handle.write(report.model_dump_json(indent=2))
    print(f"-> {STRUCTURES_REPORT}")


class StructureVerificationReport(BaseModel):
    """Which substrate structures the predictions rested on, and which survive the check."""

    model_config = ConfigDict(extra="forbid")

    built_at: str
    sources: dict[str, str]
    n_pairs_before: int
    n_distinct_structures_checked: int
    n_structures_failing: int
    n_structures_repaired: int
    n_structures_unrepairable: int
    repaired: list[dict[str, str]]
    unrepairable: list[dict[str, str]]
    n_pairs_after: int
    n_pairs_replaced: int
    n_pairs_dropped: int
    n_units_before: int
    n_units_after: int
    units_lost: list[dict[str, str]]


def stage_verify() -> None:
    """Re-resolve every substrate structure the predictions rest on, and repair or drop it.

    The pair set is taken from the predictor table that carries every pair, not rebuilt
    from the GEM, because the question is what the EXISTING predictions rest on. Each
    distinct ``(metabolite, SMILES)`` is tested against the GEM's own formula and charge.
    A failing structure is re-resolved through the two verified routes; where that
    succeeds the pair is REPLACED and has to be re-predicted, and where it does not the
    pair is DROPPED, because a value computed for a molecule whose composition disagrees
    with the GEM's is not a value for that reaction.

    Dropping is why the verified tier counts can be WORSE than the unverified ones. That is
    the point: a unit whose only substrate is an unresolvable placeholder never had a
    turnover number, it had a number computed for something else.
    """
    source = YeastGEM()
    model = source.model
    units = build_gem_tensors(model, model_dir=source.model_dir).catalytic_units
    from torchcell.metabolism.kinetics_pipeline import structure_matches_metabolite

    reference = osp.join(
        KINETICS_ROOT, REFERENCE_PREDICTOR, "processed", "k_cat.parquet"
    )
    pairs = pd.read_parquet(reference)
    distinct = pairs.drop_duplicates(["substrate_met_id", "smiles"])[
        ["substrate_met_id", "smiles"]
    ]
    failing: list[tuple[str, str]] = []
    for met_id, smiles in distinct.itertuples(index=False):
        metabolite = model.metabolites.get_by_id(met_id)
        if not structure_matches_metabolite(
            smiles, metabolite.formula or "", float(metabolite.charge or 0.0)
        ):
            failing.append((met_id, smiles))
    print(f"{len(failing)} of {len(distinct)} structures fail the composition check")

    metabolites = [model.metabolites.get_by_id(m) for m, _ in failing]
    index = CrossReferenceStructureIndex.from_metanetx(
        CHEM_PROP, CHEM_XREF, wanted_xrefs_for(metabolites)
    )
    resolver = SubstrateStructureResolver(index=index, supplied=supplied_structures())
    repaired: dict[str, ResolvedStructure] = {}
    unrepairable: list[dict[str, str]] = []
    for metabolite in metabolites:
        hit = resolver.resolve(metabolite)
        if hit is not None:
            repaired[metabolite.id] = hit
        else:
            unrepairable.append(
                {
                    "met_id": metabolite.id,
                    "name": metabolite.name or "",
                    "formula": metabolite.formula or "",
                    "n_candidates": str(len(resolver.candidates(metabolite))),
                }
            )
    print(f"repaired {len(repaired)}, unrepairable {len(unrepairable)}")

    failing_ids = {m for m, _ in failing}
    keep = pairs[~pairs["substrate_met_id"].isin(failing_ids)].copy()
    replace = pairs[pairs["substrate_met_id"].isin(repaired)].copy()
    replace["smiles"] = replace["substrate_met_id"].map(
        {k: v.smiles for k, v in repaired.items()}
    )
    replace["smiles_source"] = replace["substrate_met_id"].map(
        {k: str(v.route) for k, v in repaired.items()}
    )
    dropped = pairs[
        pairs["substrate_met_id"].isin(failing_ids - set(repaired))
    ]
    columns = [
        "unit_id",
        "reaction_id",
        "gene_id",
        "uniprot",
        "sequence_length",
        "substrate_met_id",
        "substrate_name",
        "smiles",
        "smiles_source",
        "sequence",
    ]
    verified = pd.concat([keep[columns], replace[columns]], ignore_index=True)
    # The reaction's EC numbers travel WITH the pair table because the homolog tier keys on
    # them, and the resolver in `parameters.py` has no cobra model in hand. Carrying them
    # here is what lets a substrate-matched, EC-keyed measured tier run from the mirror
    # alone, on any organism.
    ec_by_reaction = {
        reaction.id: ";".join(
            str(c)
            for c in (
                (reaction.annotation or {}).get("ec-code")
                if isinstance((reaction.annotation or {}).get("ec-code"), list)
                else [(reaction.annotation or {}).get("ec-code")]
            )
            if c
        )
        for reaction in model.reactions
    }
    verified["reaction_ec"] = verified["reaction_id"].map(ec_by_reaction)
    verified = verified.sort_values(["unit_id", "gene_id", "substrate_met_id"])
    verified = verified.reset_index(drop=True)
    verified.to_parquet(VERIFIED_PAIRS, index=False)
    # THE SUBSTRATE MAP IS WIDER THAN THE PAIR TABLE, deliberately. The pair table is what
    # the predictors were run on; the substrate map is what a database row is matched
    # against, and there is no reason to withhold a structure from the match just because no
    # predictor was run on it. Every substrate of every unit's reaction is resolved through
    # the same formula-checked routes, which is also what the portable pipeline does, so the
    # in-place resolver and the pipeline see the same substrates and report the same tiers.
    full_index = CrossReferenceStructureIndex.from_metanetx(
        CHEM_PROP, CHEM_XREF, wanted_xrefs_for(list(model.metabolites))
    )
    db_dir = osp.join(source.model_dir, "data", "databases")
    full_resolver = SubstrateStructureResolver(
        name_table=read_smiles_db(osp.join(db_dir, "smilesDB.tsv")),
        index=full_index,
        supplied=supplied_structures(),
    )
    structures: dict[str, Any] = {}
    substrate_rows: list[dict[str, Any]] = []
    for unit_id in range(units.n_units):
        reaction = model.reactions[int(units.unit_reaction[unit_id])]
        for metabolite, coefficient in reaction.metabolites.items():
            if coefficient >= 0:
                continue
            if metabolite.id not in structures:
                structures[metabolite.id] = full_resolver.resolve(metabolite)
            # A substrate with no verified structure still gets a row, with an empty
            # SMILES. Its NAME is still a correct identification of what the reaction
            # consumes, and a database row that names the same substrate is matched on
            # that; only the structure-based key is withheld, because there is none.
            hit = structures[metabolite.id]
            substrate_rows.append(
                {
                    "unit_id": unit_id,
                    "reaction_id": reaction.id,
                    "substrate_met_id": metabolite.id,
                    "substrate_name": metabolite.name or "",
                    "smiles": hit.smiles if hit is not None else "",
                    "smiles_source": str(hit.route) if hit is not None else "none",
                    "reaction_ec": ec_by_reaction.get(reaction.id, ""),
                }
            )
    substrate_map = pd.DataFrame(substrate_rows)
    feature_path = osp.join(KINETICS_ROOT, VERIFIED_PAIRS_FEATURE)
    os.makedirs(osp.dirname(feature_path), exist_ok=True)
    write_parquet_atomically(substrate_map, feature_path)
    substrate_map.to_parquet(
        osp.join(RESULTS_DIR, "kinetic_verified_substrates.parquet"), index=False
    )
    print(
        f"substrate map: {len(substrate_map)} rows, "
        f"{substrate_map['substrate_met_id'].nunique()} metabolites "
        f"({int((substrate_map['smiles'] != '').sum())} rows with a structure), "
        f"{substrate_map['unit_id'].nunique()} units -> {feature_path}"
    )

    units_before = set(pairs["unit_id"].unique())
    units_after = set(verified["unit_id"].unique())
    lost = sorted(units_before - units_after)
    report = StructureVerificationReport(
        built_at=datetime.now(UTC).isoformat(),
        sources={
            reference: sha256_of(reference),
            **index.source_paths,
            LIPID_STRUCTURES: sha256_of(LIPID_STRUCTURES),
            KINETIC_STRUCTURES: sha256_of(KINETIC_STRUCTURES),
        },
        n_pairs_before=int(len(pairs)),
        n_distinct_structures_checked=int(len(distinct)),
        n_structures_failing=len(failing),
        n_structures_repaired=len(repaired),
        n_structures_unrepairable=len(unrepairable),
        repaired=[
            {
                "met_id": k,
                "name": model.metabolites.get_by_id(k).name or "",
                "formula": model.metabolites.get_by_id(k).formula or "",
                "route": str(v.route),
                "key": v.key,
                "smiles": v.smiles,
            }
            for k, v in sorted(repaired.items())
        ],
        unrepairable=unrepairable,
        n_pairs_after=int(len(verified)),
        n_pairs_replaced=int(len(replace)),
        n_pairs_dropped=int(len(dropped)),
        n_units_before=len(units_before),
        n_units_after=len(units_after),
        units_lost=[
            {
                "unit_id": str(u),
                "reaction_id": str(
                    pairs[pairs["unit_id"] == u]["reaction_id"].iloc[0]
                ),
                "substrates": "; ".join(
                    sorted(set(pairs[pairs["unit_id"] == u]["substrate_name"]))
                ),
            }
            for u in lost
        ],
    )
    with open(VERIFICATION_REPORT, "w") as handle:
        handle.write(report.model_dump_json(indent=2))
    print(
        f"verified pairs {len(verified)} over {len(units_after)} units "
        f"({len(lost)} units lost, {len(dropped)} pairs dropped, "
        f"{len(replace)} pairs re-pointed)"
    )
    print(f"-> {VERIFIED_PAIRS}\n-> {VERIFICATION_REPORT}")


def stage_predict_verified(predictor: str, device: str, block_size: int) -> None:
    """Bring one predictor's table up to the verified pair set, then write its version.

    A prediction depends only on ``(sequence, SMILES)``, so a pair whose exact
    ``(gene, SMILES)`` the predictor has already scored is REUSED rather than re-run. That
    is exact reuse, not an approximation, and it is what keeps the GPU cost proportional to
    what actually changed.
    """
    import time

    started = time.monotonic()
    wanted = pd.read_parquet(VERIFIED_PAIRS)
    processed = osp.join(KINETICS_ROOT, predictor, "processed")
    parameters = [p for p, names in ALL_PREDICTORS.items() if predictor in names]

    existing = pd.read_parquet(osp.join(processed, f"{parameters[0]}.parquet"))
    have = set(zip(existing["gene_id"], existing["smiles"]))
    todo = wanted[
        ~pd.Series(list(zip(wanted["gene_id"], wanted["smiles"])), index=wanted.index).isin(
            have
        )
    ]
    print(f"{predictor}: {len(wanted)} wanted, {len(todo)} to predict")

    fresh = pd.DataFrame()
    if len(todo):
        fresh = run_predictor(predictor, todo, device, block_size)
    write_verified_tables(
        predictor, wanted, fresh, parameters, runtime_seconds=time.monotonic() - started
    )


def run_predictor(
    predictor: str, rows: pd.DataFrame, device: str, block_size: int
) -> pd.DataFrame:
    """Score one block of pairs with one predictor, returning the rows plus its columns."""
    from torchcell.datasets.kinetics import KINETICS_CONFIGS

    if predictor == "ezkinetics":
        import subprocess
        import tempfile

        runner = osp.join(osp.dirname(osp.abspath(__file__)), "run_ezkinetics_pinned.py")
        out = rows.copy()
        with tempfile.TemporaryDirectory() as work:
            inputs = osp.join(work, "inputs.parquet")
            rows.to_parquet(inputs, index=False)
            for parameter in ("k_cat", "K_M"):
                out_path = osp.join(work, f"{parameter}.parquet")
                subprocess.run(
                    [
                        sys.executable, runner,
                        "--parameter", parameter,
                        "--inputs", inputs,
                        "--output", out_path,
                        "--device", device,
                    ],
                    check=True,
                )
                scored = pd.read_parquet(out_path)
                out[parameter] = scored[parameter].to_numpy()
                out["weights_sha256"] = scored["weights_sha256"].to_numpy()
        return out
    dataset = KINETICS_CONFIGS[predictor](
        root=KINETICS_ROOT, inputs=rows, data_root=DATA_ROOT, device=device
    )
    blocks = [
        dataset.predict_batch(rows.iloc[start : start + block_size])
        for start in range(0, len(rows), block_size)
    ]
    out = rows.join(pd.concat(blocks))
    out["weights_sha256"] = dataset._weights_sha256
    return out


def write_verified_tables(
    predictor: str,
    wanted: pd.DataFrame,
    fresh: pd.DataFrame,
    parameters: list[str],
    runtime_seconds: float | None = None,
) -> None:
    """Assemble ``processed/verified/<parameter>.parquet`` and name what it supersedes.

    Reused values come from the unverified table by ``(gene, SMILES)``; new ones from this
    run. The unverified table is never modified, so both versions stay on disk and the
    manifest pins the sha256 of the one this replaces.
    """
    out_dir = osp.join(KINETICS_ROOT, predictor, VERIFIED_SUBDIR)
    os.makedirs(out_dir, exist_ok=True)
    for parameter in parameters:
        previous = osp.join(KINETICS_ROOT, predictor, "processed", f"{parameter}.parquet")
        existing = pd.read_parquet(previous)
        reuse = (
            existing.drop_duplicates(["gene_id", "smiles"])
            .set_index(["gene_id", "smiles"])[parameter]
            .to_dict()
        )
        if len(fresh) and parameter in fresh:
            reuse.update(
                fresh.drop_duplicates(["gene_id", "smiles"])
                .set_index(["gene_id", "smiles"])[parameter]
                .to_dict()
            )
        table = wanted.copy()
        table[parameter] = [
            reuse.get((g, s)) for g, s in zip(table["gene_id"], table["smiles"])
        ]
        table["predictor"] = predictor
        table["provenance"] = "predicted"
        weights = (
            str(fresh["weights_sha256"].iloc[0])
            if len(fresh) and "weights_sha256" in fresh
            else str(existing["weights_sha256"].iloc[0])
        )
        table["weights_sha256"] = weights
        priced = table[table[parameter].notna()]
        out_path = osp.join(out_dir, f"{parameter}.parquet")
        write_parquet_atomically(priced, out_path)
        manifest = {
            "predictor": predictor,
            "parameter": parameter,
            "version": VERIFIED_SUBDIR,
            "built_at": datetime.now(UTC).isoformat(),
            "supersedes_path": previous,
            "supersedes_sha256": sha256_of(previous),
            "verified_pairs_path": VERIFIED_PAIRS,
            "verified_pairs_sha256": sha256_of(VERIFIED_PAIRS),
            "n_rows": int(len(priced)),
            "n_rows_superseded": int(len(existing)),
            "n_rows_newly_predicted": int(len(fresh)),
            "n_units": int(priced["unit_id"].nunique()),
            "n_units_superseded": int(existing["unit_id"].nunique()),
            "n_pairs_unpriced": int(len(table) - len(priced)),
            "weights_sha256": weights,
            "runtime_seconds": (
                None if runtime_seconds is None else round(runtime_seconds, 1)
            ),
        }
        with open(osp.join(out_dir, f"{parameter}_manifest.json"), "w") as handle:
            json.dump(manifest, handle, indent=2)
        print(json.dumps(manifest, indent=2))


def stage_versions() -> None:
    """Write the table-version lineage: every parquet, its sha256 and its unit count.

    A results JSON records ``kcat_tier_counts``, and those counts are the fingerprint of a
    table version. This file is what turns that fingerprint back into an exact set of
    parquets, so a reader can tell which tables any given run was built on without
    re-deriving them.
    """
    versions: list[dict[str, Any]] = []
    for predictor, parameters in (
        (name, [p for p, names in ALL_PREDICTORS.items() if name in names])
        for name in sorted({n for names in ALL_PREDICTORS.values() for n in names})
    ):
        for parameter in parameters:
            for version in ("processed", VERIFIED_SUBDIR):
                path = osp.join(KINETICS_ROOT, predictor, version, f"{parameter}.parquet")
                if not osp.exists(path):
                    continue
                frame = pd.read_parquet(path, columns=["unit_id"])
                versions.append(
                    {
                        "predictor": predictor,
                        "parameter": parameter,
                        "version": version,
                        "path": path,
                        "sha256": sha256_of(path),
                        "n_rows": int(len(frame)),
                        "n_units": int(frame["unit_id"].nunique()),
                    }
                )
    feature = osp.join(KINETICS_ROOT, VERIFIED_PAIRS_FEATURE)
    out = {
        "built_at": datetime.now(UTC).isoformat(),
        "substrate_map": {"path": feature, "sha256": sha256_of(feature)},
        "verified_pairs": {"path": VERIFIED_PAIRS, "sha256": sha256_of(VERIFIED_PAIRS)},
        "tables": versions,
    }
    path = osp.join(RESULTS_DIR, "kinetic_table_versions.json")
    with open(path, "w") as handle:
        json.dump(out, handle, indent=2)
    for row in versions:
        print(
            f"{row['predictor']:11} {row['parameter']:6} {row['version']:10} "
            f"units {row['n_units']:5} {row['sha256']}"
        )
    print(f"-> {path}")


def stage_predict(predictor: str, device: str, block_size: int) -> None:
    """Run one predictor over the new rows only and append to its parquet.

    Appending rather than rebuilding is what keeps the 3,552 already-predicted units
    byte-identical: a rebuild would re-run every pair, and a predictor that is not
    bit-reproducible across hardware would move values that nothing asked to move.
    """
    import time

    from torchcell.datasets.kinetics import KINETICS_CONFIGS

    started = time.monotonic()
    pairs = pd.read_parquet(NEW_PAIRS)
    if predictor == "ezkinetics":
        # EZKinetics is a standalone runner rather than a registered dataset, because its
        # two parameters use two different protein encoders and it is invoked once per
        # parameter. Each invocation writes the full row table with that one parameter's
        # column, so the two are appended separately.
        import subprocess
        import tempfile

        runner = osp.join(osp.dirname(osp.abspath(__file__)), "run_ezkinetics_pinned.py")
        with tempfile.TemporaryDirectory() as work:
            for parameter in ("k_cat", "K_M"):
                out_path = osp.join(work, f"{parameter}.parquet")
                subprocess.run(
                    [
                        sys.executable, runner,
                        "--parameter", parameter,
                        "--inputs", NEW_PAIRS,
                        "--output", out_path,
                        "--device", device,
                    ],
                    check=True,
                )
                append_to_parquets(
                    "ezkinetics",
                    pd.read_parquet(out_path),
                    [parameter],
                    runtime_seconds=time.monotonic() - started,
                )
        return
    dataset = KINETICS_CONFIGS[predictor](
        root=KINETICS_ROOT, inputs=pairs, data_root=DATA_ROOT, device=device
    )
    blocks = [
        dataset.predict_batch(pairs.iloc[start : start + block_size])
        for start in range(0, len(pairs), block_size)
    ]
    predictions = pd.concat(blocks)
    table = pairs.join(predictions)
    table["predictor"] = predictor
    table["weights_sha256"] = dataset._weights_sha256
    table["provenance"] = "predicted"
    append_to_parquets(
        predictor,
        table,
        [p.value for p in dataset.EMITS],
        runtime_seconds=time.monotonic() - started,
    )


def append_to_parquets(
    predictor: str,
    table: pd.DataFrame,
    parameters: list[str],
    runtime_seconds: float | None = None,
) -> None:
    """Concatenate the new rows onto each parameter's parquet and record the append."""
    processed = osp.join(KINETICS_ROOT, predictor, "processed")
    for parameter in parameters:
        path = osp.join(processed, f"{parameter}.parquet")
        existing = pd.read_parquet(path)
        fresh = table[table[parameter].notna()]
        fresh = fresh[[c for c in existing.columns if c in fresh.columns]]
        before_units = int(existing["unit_id"].nunique())
        combined = pd.concat([existing, fresh], ignore_index=True)
        combined.to_parquet(path, index=False)
        manifest = {
            "predictor": predictor,
            "parameter": parameter,
            "appended_at": datetime.now(UTC).isoformat(),
            "n_rows_before": int(len(existing)),
            "n_rows_appended": int(len(fresh)),
            "n_units_before": before_units,
            "n_units_after": int(combined["unit_id"].nunique()),
            "new_pairs_source": NEW_PAIRS,
            "new_pairs_sha256": sha256_of(NEW_PAIRS),
            "weights_sha256": str(table["weights_sha256"].iloc[0])
            if len(table)
            else "",
            # Wall clock for the whole predictor invocation, model load included. Absent
            # from the manifests written on 2026-09-10, which predate the field.
            "runtime_seconds": (
                None if runtime_seconds is None else round(runtime_seconds, 1)
            ),
        }
        with open(osp.join(processed, f"{parameter}_append_manifest.json"), "w") as h:
            json.dump(manifest, h, indent=2)
        print(json.dumps(manifest, indent=2))


#: What a remaining default unit would need, keyed by the residual reason
#: ``kinetic_coverage_gaps.py`` records for its substrates. The judgment in the third field
#: is about tooling, not about effort: "reachable" means an open-source tool can produce the
#: missing input without changing what the GEM asserts.
CEILING_ROUTES: dict[str, tuple[str, bool]] = {
    "R-group placeholder, the GEM defines the species only up to a substituent": (
        "a GEM curation decision, not a data lookup: the species has no structure until a "
        "representative acyl chain is chosen for it, and choosing one changes what the "
        "model asserts. RDKit can write the structure once the choice is made",
        False,
    ),
    "macromolecular species with no formula in the GEM": (
        "a predictor whose substrate may be a protein. Every mirrored predictor takes a "
        "small-molecule SMILES, so this is a model class we do not have rather than an "
        "input we failed to find",
        False,
    ),
    "protein-bound species named as an enzyme state, not a compound": (
        "a predictor whose substrate may be a protein, as above",
        False,
    ),
    "no candidate structure in any mirrored catalog for this compound": (
        "one more name in the acyl-chain grammar of lipid_structure_writer.py. The compound "
        "is an ordinary small molecule that no mirrored catalog happens to carry, and RDKit "
        "writes it from the name once the grammar covers it",
        True,
    ),
    "candidate structures exist but none matches the GEM's composition, so the "
    "GEM's formula is a partial or protein-bound species": (
        "a decision about what the GEM means by the species, not a lookup. The catalogs "
        "carry the compound; the GEM writes only the cofactor part of a protein-bound "
        "species, so accepting a catalog structure would silently change the substrate",
        False,
    ),
}


def stage_report() -> None:
    """Re-resolve both tables, four ways, and measure what the verification moved.

    Four resolutions, because three separate changes landed at once and a single after
    number would not say which did what:

    ``unverified_accession_key``
        what the flux layer read before any of this: predictions on unchecked structures,
        measured keyed on the accession alone.
    ``verified_accession_key``
        the structure fix alone.
    ``verified_substrate_key``
        the structure fix plus the substrate-matched measured tier.
    ``verified_substrate_key_with_homolog``
        the tier set that ships, adding the cross-organism EC tier.
    """
    import numpy as np

    source = YeastGEM()
    gem = build_gem_tensors(source.model, model_dir=source.model_dir)
    units = gem.catalytic_units
    plausibility = osp.join(EXPERIMENT_ROOT, PLAUSIBILITY_RESULT)
    substrates = SubstrateMatch.from_verified_pairs(
        osp.join(KINETICS_ROOT, VERIFIED_PAIRS_FEATURE)
    )
    empty = SubstrateMatch()
    homolog = osp.join(DATA_ROOT, "data/enzyme_kinetics/open_enzyme_database/all")
    variants = {
        "unverified_accession_key": ("processed", empty, None, True),
        "verified_accession_key": ("verified", empty, None, True),
        "verified_substrate_key": ("verified", substrates, None, False),
        "verified_substrate_key_with_homolog": ("verified", substrates, homolog, False),
    }

    out: dict[str, Any] = {"built_at": datetime.now(UTC).isoformat(), "parameters": {}}
    out["ceiling"] = coverage_ceiling(source.model)
    for parameter in ("k_cat", "K_M"):
        resolve = resolve_kcat_table if parameter == "k_cat" else resolve_km_table
        block: dict[str, Any] = {"n_units": units.n_units, "variants": {}}
        values_by_variant: dict[str, np.ndarray] = {}
        for name, (version, match, homolog_dir, legacy) in variants.items():
            predicted = load_predicted_unit_table(
                parameter, units.n_units, KINETICS_ROOT, plausibility, version=version
            )
            # The legacy variants reproduce the accession-only key, which accepted a row
            # without checking its substrate. That is what an empty SubstrateMatch refuses,
            # so it is reproduced explicitly rather than by disabling the check.
            table = resolve(
                units,
                source.model_dir,
                OED_MIRROR,
                predicted=predicted,
                substrates=_accession_only_match(KINETICS_ROOT) if legacy else match,
                homolog_mirror_dir=homolog_dir,
            )
            block["variants"][name] = {
                "predictor_version": version,
                "selected_predictors": predicted.selection.selected,
                "n_units_with_prediction": int(predicted.mask.sum()),
                "tier_counts": table.tier_counts(),
                "experimental_fraction": float(table.experimental_coverage.fraction),
                "predictor_sha256": predicted.sha256,
                "notes": table.notes,
            }
            values_by_variant[name] = table.values.numpy().copy()
            print(f"{parameter:6} {name:38} {table.tier_counts()}")

        before = values_by_variant["unverified_accession_key"]
        after = values_by_variant["verified_substrate_key_with_homolog"]
        moved = np.abs(np.log10(np.maximum(after, 1e-12) / np.maximum(before, 1e-12)))
        changed = moved > 1e-9
        block["value_movement"] = {
            "n_units_changed": int(changed.sum()),
            "n_units_moved_over_1_decade": int((moved > 1.0).sum()),
            "n_units_moved_over_2_decades": int((moved > 2.0).sum()),
            "median_decades_among_changed": float(np.median(moved[changed]))
            if changed.any()
            else 0.0,
            "max_decades": float(moved.max()),
        }
        block["structure_repair_movement"] = _structure_repair_movement(
            parameter, units.n_units, plausibility
        )
        out["parameters"][parameter] = block
        print(f"{parameter:6} movement {block['value_movement']}")
        print(f"{parameter:6} repair   {block['structure_repair_movement']}")
    with open(CLOSURE_REPORT, "w") as handle:
        json.dump(out, handle, indent=2)
    print(f"-> {CLOSURE_REPORT}")


def _accession_only_match(kinetics_root: str) -> SubstrateMatch:
    """The old accession-only key, reproduced exactly, for the before-and-after only.

    ``accept_all_unchecked`` makes every row match, which is what keying on the accession
    alone did. The EC map is kept real so the homolog tier, where a variant asks for it,
    still keys on the reaction's own chemistry.
    """
    real = SubstrateMatch.from_verified_pairs(
        osp.join(kinetics_root, VERIFIED_PAIRS_FEATURE)
    )
    return SubstrateMatch(
        ec=real.ec,
        accept_all_unchecked=True,
        source_path="accession-only reproduction, not a real substrate map",
    )


def _structure_repair_movement(
    parameter: str, n_units: int, plausibility: str
) -> dict[str, Any]:
    """How far the repaired structures moved the PREDICTED value, on the units they touch.

    Isolated from the tier change: both sides are the predicted tier, one computed on the
    unchecked structures and one on the verified ones, compared only where both exist.
    """
    import numpy as np

    before = load_predicted_unit_table(
        parameter, n_units, KINETICS_ROOT, plausibility, version="processed"
    )
    after = load_predicted_unit_table(
        parameter, n_units, KINETICS_ROOT, plausibility, version="verified"
    )
    both = (before.mask & after.mask).numpy()
    ratio = np.zeros(n_units)
    ratio[both] = np.abs(
        np.log10(after.values.numpy()[both] / before.values.numpy()[both])
    )
    changed = both & (ratio > 1e-9)
    return {
        "n_units_with_a_prediction_before": int(before.mask.sum()),
        "n_units_with_a_prediction_after": int(after.mask.sum()),
        "n_units_lost_to_unrepairable_structures": int(
            (before.mask & ~after.mask).sum()
        ),
        "n_units_whose_predicted_value_changed": int(changed.sum()),
        "n_changed_over_1_decade": int((ratio > 1.0).sum()),
        "median_decades_among_changed": float(np.median(ratio[changed]))
        if changed.any()
        else 0.0,
        "max_decades": float(ratio.max()),
    }


def coverage_ceiling(model: Any) -> dict[str, Any]:
    """Per remaining default unit: the reason, what it would take, and whether it is reachable.

    Also measures a quality bound that the tier counts hide. A unit is on the predicted tier
    if any predictor emitted a value for it, and the predictors were run on whatever SMILES
    the input builder found, WITHOUT the composition check this fix introduced. So some
    predicted units rest on a structure whose formula does not match the GEM's, and that
    count is the honest ceiling on how far the predicted tier can be trusted.
    """
    gaps_path = osp.join(RESULTS_DIR, "kinetic_coverage_gaps.json")
    with open(gaps_path) as handle:
        gaps = json.load(handle)

    # A substrate the structure writer TRIED and refused is not reachable, whatever its
    # reason class says. The reason class is an inference from the formula; the writer's
    # refusal is a measurement, and a measurement outranks an inference.
    refused: dict[str, str] = {}
    if osp.exists(KINETIC_STRUCTURES):
        with open(KINETIC_STRUCTURES) as handle:
            for entry in json.load(handle)["structures"]:
                if not entry["formula_match"]:
                    refused[entry["name"]] = (
                        f"{entry['failure']}; the writer computed "
                        f"{entry['rdkit_formula']} against the GEM's {entry['gem_formula']}"
                    )

    # ONE ROW PER (unit, substrate), not per reason. A unit with two unresolved substrates
    # carries two reasons, and attributing a reason to the unit rather than to the substrate
    # that caused it made a unit look reachable because its OTHER substrate was.
    from kinetic_coverage_gaps import residual_substrate_reason

    candidates: dict[str, int] = {}
    structures_report = osp.join(RESULTS_DIR, "kinetic_coverage_structures.json")
    if osp.exists(structures_report):
        with open(structures_report) as handle:
            for entry in json.load(handle)["unresolved"]:
                candidates[entry["met_id"]] = int(entry["n_candidates"])

    rows: list[dict[str, Any]] = []
    for unit in gaps["parameters"]["k_cat"]["default_tier_units"]:
        for met_id, name, formula in unit["unresolved_substrates"]:
            reason = residual_substrate_reason(formula, name, candidates.get(met_id))
            requirement, reachable = CEILING_ROUTES.get(
                reason, ("unclassified residual reason", False)
            )
            if name in refused:
                reachable = False
                requirement = (
                    f"the structure writer tried this substrate and refused it: "
                    f"{refused[name]}"
                )
            rows.append(
                {
                    "unit_id": unit["unit_id"],
                    "reaction_id": unit["reaction_id"],
                    "gene_ids": unit["gene_ids"],
                    "substrate_met_id": met_id,
                    "substrate": name,
                    "formula": formula,
                    "reason": reason,
                    "what_it_would_take": requirement,
                    "reachable_with_open_source_tools": reachable,
                }
            )

    # --- how much of the predicted tier rests on an unchecked structure -----------------
    # Both versions are checked, so the report says what the fix removed rather than only
    # what is left: `processed` is the table the predictions were first computed on and
    # `verified` is the one that ships.
    quality = {}
    for version in ("processed", VERIFIED_SUBDIR):
        quality[version] = _structure_quality(model, version)
    return {
        "n_default_units": len(gaps["parameters"]["k_cat"]["default_tier_units"]),
        "n_reachable_with_open_source_tools": sum(
            1 for r in rows if r["reachable_with_open_source_tools"]
        ),
        "remaining": rows,
        "structure_quality_by_version": quality,
    }


def _structure_quality(model: Any, version: str) -> dict[str, int]:
    """How many units in one version of the tables rest only on failing structures."""
    from torchcell.metabolism.kinetics_pipeline import structure_matches_metabolite

    frame = pd.read_parquet(
        osp.join(KINETICS_ROOT, "eitlem", version, "k_cat.parquet"),
        columns=["unit_id", "substrate_met_id", "smiles"],
    ).drop_duplicates(["unit_id", "substrate_met_id"])
    verdict: dict[tuple[str, str], bool] = {}
    passing: set[int] = set()
    for unit_id, met_id, smiles in frame.itertuples(index=False):
        key = (met_id, smiles)
        if key not in verdict:
            metabolite = model.metabolites.get_by_id(met_id)
            verdict[key] = structure_matches_metabolite(
                smiles, metabolite.formula or "", float(metabolite.charge or 0.0)
            )
        if verdict[key]:
            passing.add(int(unit_id))
    every = {int(u) for u in frame["unit_id"].unique()}
    return {
        "n_units_with_a_prediction": len(every),
        "n_units_where_at_least_one_substrate_passes_the_composition_check": len(passing),
        "n_units_resting_only_on_structures_that_fail_it": len(every - passing),
        "n_distinct_substrate_structures_checked": len(verdict),
        "n_that_fail": sum(1 for v in verdict.values() if not v),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        required=True,
        choices=[
            "resolve", "predict", "verify", "predict-verified", "report", "versions",
        ],
    )
    parser.add_argument("--predictor", default="eitlem")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--block-size", type=int, default=500)
    args = parser.parse_args()
    if args.stage == "resolve":
        stage_resolve()
    elif args.stage == "predict":
        stage_predict(args.predictor, args.device, args.block_size)
    elif args.stage == "verify":
        stage_verify()
    elif args.stage == "predict-verified":
        stage_predict_verified(args.predictor, args.device, args.block_size)
    elif args.stage == "versions":
        stage_versions()
    else:
        stage_report()


if __name__ == "__main__":
    main()
