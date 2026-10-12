# experiments/026-metabolism-flux/scripts/external_pka_pilot.py
# [[experiments.026-metabolism-flux.scripts.external_pka_pilot]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/026-metabolism-flux/scripts/external_pka_pilot.py

r"""Price cache-absent metabolites end to end through the external-pKa path.

The pilot set is the three betaxanthin intermediates that
``pathway_thermo_betaxanthin.json`` reports as absent, plus the sixteen metabolites whose
``final_blocker`` in ``unresolved_metabolites.json`` is ``structure_absent_from_cache``.
Each one is carried as far as it will go:

1. **Look it up again, by InChIKey.** ``pathway_thermo.py`` resolves a structure with
   ``cc.get_compound_by_inchi``, an exact string match. The cache stores the InChI of the
   MAJOR MICROSPECIES, whose protonation block differs from the neutral molecule's, so a
   compound that is present can still miss. Anything that resolves on the first two
   InChIKey blocks was never absent and needs no pKa prediction.
2. **Predict pKas** for whatever is genuinely absent, with the predictor chosen by
   ``--tool``. This step runs in the predictor's own environment and writes a pKa file.
3. **Create the compound and price it** through
   ``LocalCompoundCache(pkas_provided_externally=True)`` on ``equilibrator-assets``
   master, then ``ComponentContribution.standard_dg_formation`` plus the Legendre
   transform, giving ``dGf'0`` and its uncertainty at the study conditions.

Group decomposition is the step that no pKa predictor can help with: a compound whose
groups the decomposer does not recognize gets ``compound.group_vector = None`` and
component contribution has nothing to price. Those are reported separately from pKa
failures, because they are a different ceiling.

Run in three phases::

    # 1. resolve structures and say which are really absent (equilibrator env)
    ~/miniconda3/envs/equilibrator-ext/bin/python \
        experiments/026-metabolism-flux/scripts/external_pka_pilot.py resolve

    # 2. predict pKas for the absent ones (predictor env)
    PATH=$HOME/miniconda3/envs/pkasolver-env/bin:$PATH \
        ~/miniconda3/envs/pkasolver-env/bin/python ... predict --tool pkasolver-lite

    # 3. create + price (equilibrator-ext env, assets master)
    ~/miniconda3/envs/equilibrator-ext/bin/python ... price

Outputs ``results/external_pka_pilot.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
from datetime import datetime, timezone
from typing import cast

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

DATA_ROOT = cast(str, os.environ["DATA_ROOT"])
EXPERIMENT_ROOT = cast(str, os.environ["EXPERIMENT_ROOT"])
RESULTS = osp.join(EXPERIMENT_ROOT, "026-metabolism-flux", "results")
GEM_DIR = osp.join(DATA_ROOT, "data", "torchcell", "yeast-GEM", "yeast-GEM-9.0.2")
SMILES_DB = osp.join(GEM_DIR, "data", "databases", "smilesDB.tsv")
METANETX_DIR = osp.join(DATA_ROOT, "data", "enzyme_kinetics", "metanetx")
CHEM_PROP = osp.join(METANETX_DIR, "chem_prop.tsv")
CHEM_XREF = osp.join(METANETX_DIR, "chem_xref.tsv")

AUDIT_PATH = osp.join(RESULTS, "unresolved_metabolites.json")
BETAXANTHIN_PATH = osp.join(RESULTS, "pathway_thermo_betaxanthin.json")
RESOLVED_PATH = osp.join(RESULTS, "external_pka_pilot_resolved.json")
PKA_PATH = osp.join(RESULTS, "external_pka_pilot_pkas.json")
PILOT_PATH = osp.join(RESULTS, "external_pka_pilot.json")

# The writable copy of the compound cache. LocalCompoundCache refuses the default cache
# so that a compound created here can never contaminate the shipped one.
LOCAL_CCACHE = osp.join(DATA_ROOT, "tmp", "pka_local_ccache.sqlite")

P_H = 7.2
IONIC_STRENGTH_M = 0.25
TEMPERATURE_K = 303.15
P_MG = 3.0
MIN_PH = 0.0
MAX_PH = 14.0


class PilotTarget(BaseModel):
    """One metabolite the pilot tries to price, and where its structure came from."""

    key: str
    source: str
    name: str
    met_ids: list[str] = Field(default_factory=list)
    formula: str | None = None
    chemical_class: str | None = None
    smiles: str | None = None
    structure_source: str | None = None
    inchi_key: str | None = None
    cache_hit_compound_id: int | None = None
    cache_hit_has_group_vector: bool | None = None
    genuinely_absent: bool = False
    blocker: str | None = None


class ResolvedSet(BaseModel):
    """The pilot targets after the InChIKey re-lookup."""

    built_at: str
    n_targets: int
    n_with_structure: int
    n_already_in_cache: int
    n_genuinely_absent: int
    targets: list[PilotTarget]


class PilotPkas(BaseModel):
    """Predicted pKas for the genuinely absent targets."""

    tool: str
    built_at: str
    pkas: dict[str, list[float]]
    errors: dict[str, str] = Field(default_factory=dict)


class PricedTarget(BaseModel):
    """The end state of one pilot target."""

    key: str
    name: str
    route: str
    smiles: str | None = None
    predicted_pkas: list[float] = Field(default_factory=list)
    compound_id: int | None = None
    has_group_vector: bool | None = None
    n_microspecies: int | None = None
    dgf_prime_kj_per_mol: float | None = None
    sigma_kj_per_mol: float | None = None
    outcome: str = "not_attempted"
    detail: str | None = None


class PilotResult(BaseModel):
    """What the external-pKa route reached, target by target."""

    built_at: str
    tool: str
    conditions: dict[str, float]
    n_targets: int
    n_already_in_cache: int
    n_no_structure: int
    n_attempted_creation: int
    n_priced: int
    n_failed_group_decomposition: int
    n_failed_other: int
    targets: list[PricedTarget]


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


# -------------------------------------------------------------------------- resolve


def _read_smiles_db() -> dict[str, str]:
    out: dict[str, str] = {}
    if not osp.isfile(SMILES_DB):
        return out
    with open(SMILES_DB) as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            name, smiles = parts[0].strip(), parts[1].strip()
            if name and smiles:
                out[name.lower()] = smiles
    return out


def _read_chem_prop(wanted: set[str]) -> dict[str, tuple[str, str, str]]:
    out: dict[str, tuple[str, str, str]] = {}
    if not osp.isfile(CHEM_PROP) or not wanted:
        return out
    with open(CHEM_PROP) as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9 or parts[0] not in wanted:
                continue
            out[parts[0]] = (parts[1], parts[6], parts[8])
    return out


def _read_chem_xref(wanted: set[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    if not osp.isfile(CHEM_XREF) or not wanted:
        return out
    with open(CHEM_XREF) as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            if parts[0] in wanted:
                out[parts[0]] = parts[1]
    return out


def cmd_resolve(args: argparse.Namespace) -> None:
    """Collect the pilot targets, find a structure, and re-look-up by InChIKey."""
    from equilibrator_cache.api import create_compound_cache_from_zenodo
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")

    targets: list[PilotTarget] = []

    with open(BETAXANTHIN_PATH) as handle:
        pathway = json.load(handle)
    for row in pathway["compounds"]:
        if row.get("blocker") is None:
            continue
        targets.append(
            PilotTarget(
                key=f"betaxanthin:{row['name']}",
                source="pathway_thermo_betaxanthin",
                name=row["name"],
                formula=row.get("expected_formula"),
                smiles=row.get("smiles"),
                structure_source="pubchem_via_pathway_thermo",
            )
        )

    with open(AUDIT_PATH) as handle:
        audit = json.load(handle)
    absent = [
        m
        for m in audit["metabolites_unpriced_at_baseline"]
        if m.get("final_blocker") == "structure_absent_from_cache"
    ]
    by_name: dict[str, list[dict]] = {}
    for met in absent:
        by_name.setdefault(met["name"], []).append(met)

    smiles_db = _read_smiles_db()
    mnx_wanted: set[str] = set()
    xref_wanted: set[str] = set()
    for name, mets in by_name.items():
        annotations = mets[0].get("annotations", {})
        for accession in annotations.get("metanetx.chemical", []):
            mnx_wanted.add(accession)
        for accession in annotations.get("chebi", []):
            xref_wanted.add(f"chebi:{accession.split(':')[-1]}")
        for accession in annotations.get("kegg.compound", []):
            xref_wanted.add(f"kegg.compound:{accession}")
    xref = _read_chem_xref(xref_wanted)
    mnx_wanted |= set(xref.values())
    chem_prop = _read_chem_prop(mnx_wanted)

    for name, mets in by_name.items():
        annotations = mets[0].get("annotations", {})
        smiles = None
        structure_source = None
        for accession in annotations.get("metanetx.chemical", []):
            if accession in chem_prop and chem_prop[accession][2]:
                smiles = chem_prop[accession][2]
                structure_source = f"metanetx.chem_prop:{accession}"
                break
        if smiles is None:
            for key in ("chebi", "kegg.compound"):
                for accession in annotations.get(key, []):
                    probe = (
                        f"chebi:{accession.split(':')[-1]}"
                        if key == "chebi"
                        else f"kegg.compound:{accession}"
                    )
                    mapped = xref.get(probe)
                    if mapped and mapped in chem_prop and chem_prop[mapped][2]:
                        smiles = chem_prop[mapped][2]
                        structure_source = f"metanetx.chem_prop:{mapped} via {probe}"
                        break
                if smiles:
                    break
        if smiles is None and name.lower() in smiles_db:
            smiles = smiles_db[name.lower()]
            structure_source = "yeast-GEM smilesDB"
        targets.append(
            PilotTarget(
                key=f"audit:{name}",
                source="unresolved_metabolites",
                name=name,
                met_ids=[m["met_id"] for m in mets],
                formula=mets[0].get("formula"),
                chemical_class=mets[0].get("chemical_class"),
                smiles=smiles,
                structure_source=structure_source,
            )
        )

    ccache = create_compound_cache_from_zenodo()
    for target in targets:
        if not target.smiles:
            target.blocker = "no_structure_in_any_namespace"
            continue
        mol = Chem.MolFromSmiles(target.smiles)
        if mol is None:
            target.blocker = "structure_not_parseable"
            continue
        inchi_key = Chem.MolToInchiKey(mol)
        if not inchi_key:
            target.blocker = "inchikey_generation_failed"
            continue
        target.inchi_key = inchi_key
        hits = ccache.search_compound_by_inchi_key(inchi_key.rsplit("-", 1)[0])
        if hits:
            target.cache_hit_compound_id = hits[0].id
            target.cache_hit_has_group_vector = hits[0].group_vector is not None
            target.blocker = (
                None
                if hits[0].group_vector is not None
                else "in_cache_without_group_decomposition"
            )
        else:
            target.genuinely_absent = True

    resolved = ResolvedSet(
        built_at=_timestamp(),
        n_targets=len(targets),
        n_with_structure=sum(1 for t in targets if t.smiles),
        n_already_in_cache=sum(1 for t in targets if t.cache_hit_compound_id),
        n_genuinely_absent=sum(1 for t in targets if t.genuinely_absent),
        targets=targets,
    )
    os.makedirs(RESULTS, exist_ok=True)
    with open(RESOLVED_PATH, "w") as handle:
        handle.write(resolved.model_dump_json(indent=1))
    print(f"wrote {RESOLVED_PATH}")
    print(
        f"targets {resolved.n_targets},"
        f" with structure {resolved.n_with_structure},"
        f" already in cache {resolved.n_already_in_cache},"
        f" genuinely absent {resolved.n_genuinely_absent}"
    )
    for target in targets:
        print(
            f"  {target.key}: struct={bool(target.smiles)}"
            f" cache_id={target.cache_hit_compound_id}"
            f" absent={target.genuinely_absent} blocker={target.blocker}"
        )


def _load_resolved() -> ResolvedSet:
    with open(RESOLVED_PATH) as handle:
        return ResolvedSet.model_validate_json(handle.read())


# -------------------------------------------------------------------------- predict


def cmd_predict(args: argparse.Namespace) -> None:
    """Predict pKas for the targets that are genuinely absent from the cache."""
    resolved = _load_resolved()
    wanted = [t for t in resolved.targets if t.genuinely_absent and t.smiles]
    if args.include_cached:
        wanted = [t for t in resolved.targets if t.smiles]

    pkas: dict[str, list[float]] = {}
    errors: dict[str, str] = {}

    if args.tool == "pkasolver-lite":
        from pkasolver.query import QueryModel, calculate_microstate_pka_values
        from rdkit import Chem, RDLogger

        RDLogger.DisableLog("rdApp.*")
        model = QueryModel()
        for target in wanted:
            try:
                mol = Chem.MolFromSmiles(target.smiles)
                states = calculate_microstate_pka_values(mol, query_model=model)
                pkas[target.key] = sorted(
                    (
                        float(s.pka)
                        for s in states
                        if MIN_PH < float(s.pka) < MAX_PH
                    ),
                    reverse=True,
                )
            except Exception as exc:
                errors[target.key] = f"{type(exc).__name__}: {exc}"[:300]
    else:
        raise ValueError(f"unsupported tool {args.tool}")

    out = PilotPkas(tool=args.tool, built_at=_timestamp(), pkas=pkas, errors=errors)
    with open(PKA_PATH, "w") as handle:
        handle.write(out.model_dump_json(indent=1))
    print(f"wrote {PKA_PATH}: {len(pkas)} predicted, {len(errors)} failed")
    for key, values in pkas.items():
        print(f"  {key}: {[round(v, 2) for v in values]}")


# ---------------------------------------------------------------------------- price


def cmd_price(args: argparse.Namespace) -> None:
    """Create the absent compounds from external pKas and price them."""
    import numpy as np
    from equilibrator_api import ComponentContribution, Q_
    from equilibrator_assets.local_compound_cache import LocalCompoundCache

    resolved = _load_resolved()
    with open(PKA_PATH) as handle:
        predicted = PilotPkas.model_validate_json(handle.read())

    cc = ComponentContribution()
    cc.p_h = Q_(P_H)
    cc.ionic_strength = Q_(f"{IONIC_STRENGTH_M}M")
    cc.temperature = Q_(f"{TEMPERATURE_K}K")
    cc.p_mg = Q_(P_MG)

    local = LocalCompoundCache(pkas_provided_externally=True)
    local.load_cache(LOCAL_CCACHE)

    def _price(compound) -> tuple[float | None, float | None, str]:
        try:
            mu, sigma_fin, _ = cc.standard_dg_formation(compound)
        except Exception as exc:
            return None, None, f"standard_dg_formation failed: {exc}"[:200]
        if mu is None:
            return None, None, "component contribution returned no formation energy"
        correction = compound.transform(
            Q_(P_H), Q_(f"{IONIC_STRENGTH_M}M"), Q_(f"{TEMPERATURE_K}K"), Q_(P_MG)
        )
        return (
            float(mu) + float(correction.m_as("kJ/mol")),
            float(np.linalg.norm(sigma_fin)),
            "ok",
        )

    priced: list[PricedTarget] = []
    for target in resolved.targets:
        row = PricedTarget(key=target.key, name=target.name, route="", smiles=target.smiles)
        if target.cache_hit_compound_id is not None:
            row.route = "already_in_cache_by_inchikey"
            row.compound_id = target.cache_hit_compound_id
            row.has_group_vector = target.cache_hit_has_group_vector
            hits = cc.search_compound_by_inchi_key(target.inchi_key.rsplit("-", 1)[0])
            compound = hits[0] if hits else None
            if compound is None:
                row.outcome = "cache_hit_but_compound_not_reloadable"
            elif compound.group_vector is None:
                row.outcome = "failed_group_decomposition"
            else:
                dgf, sigma, detail = _price(compound)
                row.dgf_prime_kj_per_mol = dgf
                row.sigma_kj_per_mol = sigma
                row.n_microspecies = len(compound.microspecies)
                row.outcome = "priced" if dgf is not None else "failed_other"
                row.detail = detail
            priced.append(row)
            continue

        if not target.smiles:
            row.route = "no_structure"
            row.outcome = "no_structure_in_any_namespace"
            priced.append(row)
            continue

        pkas = predicted.pkas.get(target.key)
        if pkas is None:
            row.route = "external_pka"
            row.outcome = "pka_prediction_failed"
            row.detail = predicted.errors.get(target.key)
            priced.append(row)
            continue

        row.route = "external_pka"
        row.predicted_pkas = pkas
        try:
            results = local.get_compounds(
                [target.smiles],
                mol_format="smiles",
                specified_pkas={target.smiles: pkas},
                bypass_chemaxon=True,
            )
        except Exception as exc:
            row.outcome = "compound_creation_raised"
            row.detail = f"{type(exc).__name__}: {exc}"[:300]
            priced.append(row)
            continue

        compound = results[0].compound if results else None
        if compound is None:
            row.outcome = "failed_group_decomposition"
            row.detail = f"method={results[0].method} status={results[0].status}"
            priced.append(row)
            continue

        row.compound_id = compound.id
        row.has_group_vector = compound.group_vector is not None
        row.n_microspecies = len(compound.microspecies)
        if compound.group_vector is None:
            row.outcome = "failed_group_decomposition"
            priced.append(row)
            continue
        dgf, sigma, detail = _price(compound)
        row.dgf_prime_kj_per_mol = dgf
        row.sigma_kj_per_mol = sigma
        row.outcome = "priced" if dgf is not None else "failed_other"
        row.detail = detail
        priced.append(row)

    result = PilotResult(
        built_at=_timestamp(),
        tool=predicted.tool,
        conditions={
            "p_h": P_H,
            "ionic_strength_m": IONIC_STRENGTH_M,
            "temperature_k": TEMPERATURE_K,
            "p_mg": P_MG,
        },
        n_targets=len(priced),
        n_already_in_cache=sum(
            1 for r in priced if r.route == "already_in_cache_by_inchikey"
        ),
        n_no_structure=sum(1 for r in priced if r.route == "no_structure"),
        n_attempted_creation=sum(1 for r in priced if r.route == "external_pka"),
        n_priced=sum(1 for r in priced if r.outcome == "priced"),
        n_failed_group_decomposition=sum(
            1 for r in priced if r.outcome == "failed_group_decomposition"
        ),
        n_failed_other=sum(
            1
            for r in priced
            if r.outcome
            not in ("priced", "failed_group_decomposition", "no_structure_in_any_namespace")
        ),
        targets=priced,
    )
    with open(PILOT_PATH, "w") as handle:
        handle.write(result.model_dump_json(indent=1))
    print(f"wrote {PILOT_PATH}")
    for row in priced:
        print(
            f"  {row.key}: route={row.route} outcome={row.outcome}"
            f" dgf'={row.dgf_prime_kj_per_mol} sigma={row.sigma_kj_per_mol}"
        )


# ---------------------------------------------------------------------------- reach


class ReachCount(BaseModel):
    """How far the external-pKa route can go across the unpriced metabolites."""

    built_at: str
    n_remaining_unpriced: int
    n_distinct_names: int
    tier_a_structure_exists_today: int
    tier_a_distinct_names: int
    tier_b_structure_writable_from_name: int
    tier_b_distinct_names: int
    tier_c_no_structure_can_exist: int
    tier_c_distinct_names: int
    tier_d_blocked_by_group_decomposition: int
    tier_d_distinct_names: int
    by_blocker: dict[str, int]
    tier_b_by_class: dict[str, int]
    tier_b_name_pattern_matches: int
    notes: dict[str, str]


ACYL_PATTERN = r"\d{1,2}:\d"


def cmd_reach(args: argparse.Namespace) -> None:
    """Count what the external-pKa route can and cannot reach, by blocker."""
    import re
    from collections import Counter

    with open(AUDIT_PATH) as handle:
        audit = json.load(handle)
    mets = audit["metabolites_unpriced_at_baseline"]
    remaining = [m for m in mets if not m.get("recovered_priceable")]
    blockers = Counter(m.get("final_blocker") for m in remaining)

    def _names(rows: list[dict]) -> int:
        return len({r["name"] for r in rows})

    tier_a = [m for m in remaining if m.get("final_blocker") == "structure_absent_from_cache"]
    tier_d = [
        m
        for m in remaining
        if m.get("final_blocker") == "structure_known_but_no_group_decomposition"
    ]
    tier_b = [
        m for m in remaining if m.get("final_blocker") == "no_usable_structure_available"
    ]
    tier_c = [
        m
        for m in remaining
        if m.get("final_blocker") == "placeholder_formula_no_definite_structure"
    ]
    tier_b_classes = Counter(m.get("chemical_class") for m in tier_b)
    tier_b_named = sum(1 for m in tier_b if re.search(ACYL_PATTERN, m["name"]))

    result = ReachCount(
        built_at=_timestamp(),
        n_remaining_unpriced=len(remaining),
        n_distinct_names=_names(remaining),
        tier_a_structure_exists_today=len(tier_a),
        tier_a_distinct_names=_names(tier_a),
        tier_b_structure_writable_from_name=len(tier_b),
        tier_b_distinct_names=_names(tier_b),
        tier_c_no_structure_can_exist=len(tier_c),
        tier_c_distinct_names=_names(tier_c),
        tier_d_blocked_by_group_decomposition=len(tier_d),
        tier_d_distinct_names=_names(tier_d),
        by_blocker={str(k): v for k, v in blockers.items()},
        tier_b_by_class={str(k): v for k, v in tier_b_classes.items()},
        tier_b_name_pattern_matches=tier_b_named,
        notes={
            "tier_a": (
                "a structure resolves from an existing namespace, so a pKa predictor is"
                " the only missing input"
            ),
            "tier_b": (
                "definite formula but no structure in any namespace, dominated by lipid"
                " species whose acyl composition is written in the name; a structure"
                " writer would have to compose the headgroup with the named chains and"
                " check the formula it produces against the GEM's"
            ),
            "tier_c": (
                "R, X or * in the formula, so no single structure exists and neither a"
                " pKa predictor nor a structure writer changes anything"
            ),
            "tier_d": (
                "a structure is already in the cache and the group decomposer refuses"
                " it, so external pKas do not help"
            ),
        },
    )
    path = osp.join(RESULTS, "external_pka_reach.json")
    with open(path, "w") as handle:
        handle.write(result.model_dump_json(indent=1))
    print(f"wrote {path}")
    print(result.model_dump_json(indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_resolve = sub.add_parser("resolve", help="find structures and re-look-up the cache")
    p_resolve.set_defaults(func=cmd_resolve)

    p_predict = sub.add_parser("predict", help="predict pKas for the absent targets")
    p_predict.add_argument("--tool", default="pkasolver-lite")
    p_predict.add_argument("--include-cached", action="store_true")
    p_predict.set_defaults(func=cmd_predict)

    p_price = sub.add_parser("price", help="create the compounds and price them")
    p_price.set_defaults(func=cmd_price)

    p_reach = sub.add_parser("reach", help="count what the route can reach")
    p_reach.set_defaults(func=cmd_reach)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
