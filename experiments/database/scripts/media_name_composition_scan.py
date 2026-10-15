# experiments/database/scripts/media_name_composition_scan.py
# [[experiments.database.scripts.media_name_composition_scan]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/database/scripts/media_name_composition_scan
"""Which stored media state composition only inside ``Media.name``?

Lane X of the ontology unification. Lane W measured that YeastPhenome states a medium's
dropout, carbon source and buffer only in ``Media.name``, which ``media_identity`` drops,
so six released tokens reached one served environment node. This script answers two
questions with numbers:

1. ``--yeastphenome`` -- what the pinned release actually states, per released media
   token: how it is spelled, how many columns and records it covers, what the loader
   stores for it, and which media identity it lands on. ``--prefix`` re-runs the same
   pass with the PRE-lane X of the ontology unification medium (the token truncated at '+', no typed composition), so
   the before/after table regenerates from one script. ``--l1`` re-keys every record the
   loader builds under ``environment_response._pair_key`` in both states, which is the
   number ``ENVIRONMENT_RESPONSE_DATASETS`` declares.
2. ``--library`` / ``--stores`` -- is it one dataset's bug? ``--library`` runs a detector
   over every ``MEDIA_LIBRARY`` object (the shared media every loader draws from);
   ``--stores`` runs it over the dev stores, reading each store read-only and sampling
   its keys on a stride, so a 30M-record store costs the same as a small one. The
   detector flags a medium whose NAME states something (a dropout marker, a carbon
   source, a buffer, a pH, a supplement, a dose) that its own typed fields do not carry.

Read-only on every store. Results:
``experiments/database/results/media_name_composition_scan.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import pickle
import re
from collections import Counter, defaultdict
from typing import Any

import lmdb
from dotenv import load_dotenv

from torchcell.datamodels.identity import identity_sha256, media_identity
from torchcell.datamodels.media import MEDIA_LIBRARY
from torchcell.datamodels.schema import Environment, MediaComponentRole

RESULTS = osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), "results")

# --------------------------------------------------------------------------- #
# The detector: a name that states what the typed fields do not carry
# --------------------------------------------------------------------------- #
#: Words that mark a nutrient as deliberately absent. A name carrying one while
#: ``dropouts`` and ``dropout_roles`` are both empty is stating an omission the record
#: does not carry.
_DROPOUT_MARKERS = (
    r"\bdrop-?out\b",
    r"\bminus\b",
    r"\bw/o\b",
    r"\bwithout\b",
    r"\bno\s+(carbon|nitrogen|phosph|sulf|amino)",
    r"-\s?(Ade|Arg|His|Leu|Lys|Met|Trp|Tyr|Ura|N|C)\b",
    r"\(-[A-Z]\)",
    r"\b(carbon|nitrogen|phosphate|sulfate)-free\b",
    r"\bfree\b",
)
#: Carbon sources a medium name spells out. A name carrying one with no
#: ``carbon_source`` component is stating the carbon source only in the name.
_CARBON_WORDS = (
    "glucose",
    "dextrose",
    "glycerol",
    "galactose",
    "ethanol",
    "lactate",
    "maltose",
    "mannose",
    "raffinose",
    "sucrose",
    "trehalose",
    "xylose",
    "fructose",
    "acetate",
    "arabinose",
    "pyruvate",
    "succinate",
    "citrate",
    "gluconate",
)
#: Buffers a medium name spells out, and pH, which is a buffered statement too.
_BUFFER_WORDS = ("hepes", "mops", "mes ", "tris", "phosphate buffer", "bis-tris")
_PH_RE = re.compile(r"\bph\s*[=:]?\s*[0-9]", re.IGNORECASE)
#: A dose written into the name.
_DOSE_RE = re.compile(
    r"\[[0-9.]+\s*[^\]]+\]|\b[0-9.]+\s*(%|g/L|mg/L|mM|uM|ug/ml|M)\b", re.IGNORECASE
)
#: Supplement markers.
#: ``supplementary`` is a table reference, not a medium statement, so the word form is
#: anchored to the medium sense ("supplemented with").
_SUPPLEMENT_RE = re.compile(
    r"\+\s?(ade|arg|his|leu|lys|met|trp|tyr|ura)\b|\bsupplemented\b", re.IGNORECASE
)

_CARBON_ROLES = {MediaComponentRole.carbon_source}
_SUPPLEMENT_ROLES = {MediaComponentRole.amino_acid, MediaComponentRole.nucleobase}


def _component_roles(media: dict[str, Any]) -> set[str]:
    return {
        str(c.get("role")) for c in media.get("components") or [] if isinstance(c, dict)
    }


def _environment_factors(environment: dict[str, Any] | None) -> set[str]:
    """Physical factors the ENVIRONMENT states, which carry a medium-name statement.

    A carbon source, a nitrogen source or a pH may be stated beside the medium as an
    ``EnvironmentPhysicalPerturbation`` rather than inside it (Wetmore 2015, Schmidt
    2016, every carbon-source series), so the fact is carried and the name restates it.
    """
    if not isinstance(environment, dict):
        return set()
    return {
        str(p.get("factor"))
        for p in environment.get("perturbations") or []
        if isinstance(p, dict) and p.get("factor")
    }


def _composition_is_listed(media: dict[str, Any]) -> bool:
    """Is this medium's composition entered, so an omission can be read off it?

    A medium with a listed composition states "no carbon source" by having no component
    in that role, which is how every ``M9 ... noCarbon`` object in the library says it.
    An EMPTY list cannot: ``components=[]`` already means "composition not yet entered",
    and neither can a list that is only a whole-medium deferred mixture, which asserts
    nothing about any one role.
    """
    components = [c for c in media.get("components") or [] if isinstance(c, dict)]
    if not components:
        return False
    return any(
        c.get("definition") != "composition_deferred"
        or c.get("role") != MediaComponentRole.other.value
        for c in components
    )


def _has_reduced_component(media: dict[str, Any]) -> bool:
    """A component the source REDUCED rather than removed (the partial-dropout form)."""
    for component in media.get("components") or []:
        if not isinstance(component, dict):
            continue
        concentration = component.get("concentration")
        if (
            isinstance(concentration, dict)
            and concentration.get("basis") == "reduced_from_standard"
        ):
            return True
    return False


def _component_text(media: dict[str, Any]) -> str:
    parts: list[str] = []
    for component in media.get("components") or []:
        if not isinstance(component, dict):
            continue
        compound = component.get("compound")
        if isinstance(compound, dict) and compound.get("name"):
            parts.append(str(compound["name"]).lower())
    for dropout in media.get("dropouts") or []:
        if isinstance(dropout, dict) and dropout.get("name"):
            parts.append(str(dropout["name"]).lower())
    return " ".join(parts)


def _base_composition(media: dict[str, Any]) -> tuple[set[str], str]:
    """The roles and compound names of the LIBRARY object this medium's base names.

    ``base_medium`` is a key of ``MEDIA_LIBRARY``, and the library holds a base to be a
    component SUBSET of everything deriving from it, so a derived medium that lists only
    its own additions still reaches the base's composition through that key (Baba 2006's
    "0.4% glucose MOPS medium" lists glucose, phosphate and kanamycin, and the MOPS
    buffer its name states is a component of ``MOPS_MINIMAL``). The key is a STRING in
    the record, so ``media_identity`` does not read the composition behind it; a
    statement carried only this way is reported separately rather than ignored.
    """
    key = media.get("base_medium")
    base = MEDIA_LIBRARY.get(str(key)) if key else None
    if base is None:
        return set(), ""
    dumped = base.model_dump()
    return _component_roles(dumped), _component_text(dumped)


def unread_statements(
    media: dict[str, Any], environment: dict[str, Any] | None = None
) -> list[str]:
    """Statements this medium's NAME makes that no typed field of the record carries.

    Deliberately conservative, so that what it flags is a defect and not a wording. A
    statement counts as CARRIED when any of these holds, and the name is then only
    restating it:

    * a typed field on the medium says it (``dropouts``, ``dropout_roles``, a component
      in the role, a component whose dose is a typed ``reduced_from_standard``);
    * the compound the name mentions is a component or a dropout, whatever role it was
      filed under, since a name is free text and the compound is the fact;
    * the marker is part of a REAGENT's name ("yeast nitrogen base w/o amino acids"),
      which is a component, not a medium-level omission;
    * the composition is LISTED and no component fills the role the name says is absent,
      which is how every ``M9 ... noCarbon`` object states it;
    * the ENVIRONMENT states it beside the medium as a physical factor (a carbon source,
      a nitrogen source, a pH), which is where a carbon-source series puts it.
    """
    name = str(media.get("name") or "")
    if not name:
        return []
    lowered = name.lower()
    carried = _component_text(media)
    roles = _component_roles(media)
    factors = _environment_factors(environment)
    base_roles, base_carried = _base_composition(media)
    listed = _composition_is_listed(media)
    found: list[str] = []
    if not (
        media.get("dropouts")
        or media.get("dropout_roles")
        or _has_reduced_component(media)
        or listed
    ):
        for pattern in _DROPOUT_MARKERS:
            match = re.search(pattern, name, re.IGNORECASE)
            if match is None:
                continue
            text = match.group(0).strip().lower()
            if text in carried:
                continue  # part of a reagent's own name, and that reagent is a component
            # "no carbon source" beside an EnvironmentPhysicalPerturbation that names the
            # carbon source IS carried: the record says where the carbon comes from, which
            # is the carbon-source-series design (Wetmore 2015, Banerjee 2025)
            if (
                "carbon" in text and MediaComponentRole.carbon_source.value in factors
            ) or (
                "nitrogen" in text
                and MediaComponentRole.nitrogen_source.value in factors
            ):
                continue
            found.append(f"dropout_marker:{text}")
            break
    for word in _CARBON_WORDS:
        if (
            word in lowered
            and word not in carried
            and "carbon_source" not in roles
            and MediaComponentRole.carbon_source.value not in factors
        ):
            base = "carbon_source" in base_roles or word in base_carried
            found.append(f"carbon_source:{word}" + (" (base_medium)" if base else ""))
            break
    for word in _BUFFER_WORDS:
        if word in lowered and "buffer" not in roles and word.strip() not in carried:
            base = "buffer" in base_roles
            found.append(f"buffer:{word.strip()}" + (" (base_medium)" if base else ""))
            break
    if _PH_RE.search(name) and "buffer" not in roles and "pH" not in factors:
        found.append("ph_in_name")
    if _SUPPLEMENT_RE.search(name) and not (
        roles & {r.value for r in _SUPPLEMENT_ROLES}
    ):
        found.append("supplement_in_name")
    if _DOSE_RE.search(name) and not any(
        isinstance(c, dict) and c.get("concentration")
        for c in media.get("components") or []
    ):
        found.append("dose_in_name")
    return found


# --------------------------------------------------------------------------- #
# The YeastPhenome release, per token
# --------------------------------------------------------------------------- #
def _yeastphenome_rows(raw_dir: str, *, prefix: bool) -> dict[str, Any]:
    """Per released media token: columns, records, what is stored, which identity."""
    from torchcell.datasets.scerevisiae.yeastphenome import (
        SCREENS,
        _column_meta,
        _parse_conditions,
        _parse_media,
    )

    columns: Counter[str] = Counter()
    records: Counter[str] = Counter()
    dropped: Counter[str] = Counter()
    stored: dict[str, set[str]] = defaultdict(set)
    identities: dict[str, set[str]] = defaultdict(set)
    by_stored_name: dict[str, Counter[str]] = defaultdict(Counter)
    total = 0
    for screen in SCREENS:
        path = osp.join(raw_dir, f"{screen['pmid']}_{screen['stem']}_valuez.txt")
        lines = open(path, encoding="utf-8", errors="replace").read().splitlines()
        header = lines[0].split("\t")
        rows = [ln.split("\t") for ln in lines[1:] if ln.strip()]
        for col in range(1, len(header)):
            if not header[col].strip():
                continue
            meta = _column_meta(header[col])
            if meta is None:
                continue
            token = meta["media"] if meta["media"] is not None else "<no media field>"
            columns[token] += 1
            if "growth" not in (meta["phenotype"] or "").lower():
                dropped[f"{token}::not_growth"] += 1
                continue
            if _parse_conditions(meta["condition"] or "").unrepresentable:
                dropped[f"{token}::condition_not_representable"] += 1
                continue
            n_cells = sum(
                1 for cells in rows if col < len(cells) and cells[col].strip()
            )
            media = _parse_media(meta["media"], meta["phenotype"] or "", screen)
            if prefix and meta["media"] is not None:
                media = media.model_copy(
                    update={
                        "name": meta["media"].split("+")[0].strip(),
                        "components": [],
                        "dropouts": [],
                        "dropout_roles": [],
                    }
                )
            records[token] += n_cells
            total += n_cells
            stored[token].add(
                json.dumps(
                    {
                        "name": media.name,
                        "state": media.state,
                        "is_synthetic": media.is_synthetic,
                        "base_medium": media.base_medium,
                        "components": [
                            (c.compound.name, c.role.value) for c in media.components
                        ],
                        "dropouts": [c.name for c in media.dropouts],
                        "dropout_roles": [r.value for r in media.dropout_roles],
                    },
                    sort_keys=True,
                )
            )
            identities[token].add(identity_sha256(media_identity(media))[:12])
            by_stored_name[str(media.name)][token] += n_cells
    per_identity: Counter[str] = Counter()
    identity_tokens: dict[str, set[str]] = defaultdict(set)
    for token, ids in identities.items():
        for ident in ids:
            per_identity[ident] += records[token]
            identity_tokens[ident].add(token)
    return {
        "records_total": total,
        "released_tokens": len(columns),
        "distinct_stored_names": len(by_stored_name),
        "distinct_media_identities": len(per_identity),
        "per_token": {
            token: {
                "columns": columns[token],
                "records": records[token],
                "stored": sorted(stored[token]),
                "media_identities": sorted(identities[token]),
            }
            for token in sorted(records, key=lambda t: -records[t])
        },
        "per_stored_name": {
            name: {
                "records": sum(counts.values()),
                "released_tokens": dict(counts.most_common()),
            }
            for name, counts in sorted(
                by_stored_name.items(), key=lambda kv: -sum(kv[1].values())
            )
        },
        "per_media_identity": {
            ident: {
                "records": per_identity[ident],
                "released_tokens": sorted(identity_tokens[ident]),
            }
            for ident in sorted(per_identity, key=lambda i: -per_identity[i])
        },
        "columns_not_stored": dict(dropped.most_common()),
    }


def _yeastphenome_environments(raw_dir: str, *, prefix: bool) -> dict[str, Any]:
    """Served environment identities, and how many records sit on a shared one."""
    from torchcell.datamodels.identity import environment_identity
    from torchcell.datasets.scerevisiae.yeastphenome import (
        SCREENS,
        _column_meta,
        _gap,
        _parse_conditions,
        _parse_media,
    )

    records: Counter[str] = Counter()
    tokens: dict[str, set[str]] = defaultdict(set)
    for screen in SCREENS:
        path = osp.join(raw_dir, f"{screen['pmid']}_{screen['stem']}_valuez.txt")
        lines = open(path, encoding="utf-8", errors="replace").read().splitlines()
        header = lines[0].split("\t")
        rows = [ln.split("\t") for ln in lines[1:] if ln.strip()]
        for col in range(1, len(header)):
            if not header[col].strip():
                continue
            meta = _column_meta(header[col])
            if meta is None or "growth" not in (meta["phenotype"] or "").lower():
                continue
            parsed = _parse_conditions(meta["condition"] or "")
            if parsed.unrepresentable:
                continue
            media = _parse_media(meta["media"], meta["phenotype"] or "", screen)
            if prefix and meta["media"] is not None:
                media = media.model_copy(
                    update={
                        "name": meta["media"].split("+")[0].strip(),
                        "components": [],
                        "dropouts": [],
                        "dropout_roles": [],
                    }
                )
            environment = Environment(
                media=media,
                temperature=parsed.temperature,
                perturbations=parsed.perturbations,
                aerobicity=parsed.aerobicity,
                duration_hours=parsed.duration_hours,
                duration_generations=parsed.duration_generations,
                provenance_gaps=(
                    [] if parsed.temperature is not None else [_gap("temperature")]
                ),
            )
            ident = identity_sha256(environment_identity(environment))[:12]
            n_cells = sum(
                1 for cells in rows if col < len(cells) and cells[col].strip()
            )
            records[ident] += n_cells
            tokens[ident].add(f"{meta['media']} [{media.state}]")
    shared = {i: sorted(t) for i, t in tokens.items() if len(t) > 1}
    return {
        "environment_identities": len(records),
        "identities_reached_from_several_tokens": len(shared),
        "records_on_a_shared_environment_node": sum(records[i] for i in shared),
        "shared": {
            i: {"records": records[i], "released_tokens": t}
            for i, t in sorted(shared.items(), key=lambda kv: -records[kv[0]])
        },
    }


def _yeastphenome_l1(raw_dir: str, *, prefix: bool) -> dict[str, int]:
    """Re-key every record the loader builds under the real L1 pair key."""
    from torchcell.datasets.scerevisiae.yeastphenome import (
        NPV_UNITS,
        SCREENS,
        YeastPhenomeDataset,
        _column_meta,
        _gap,
        _parse_conditions,
        _parse_media,
        _readout_method,
    )
    from torchcell.verification.environment_response import (
        _condition_signature,
        _genotype_signature,
        _study_key,
    )

    dataset = YeastPhenomeDataset.__new__(YeastPhenomeDataset)
    dataset.__dict__["name"] = "yeastphenome"
    seen: set[tuple[Any, ...]] = set()
    total = 0
    for screen in SCREENS:
        path = osp.join(raw_dir, f"{screen['pmid']}_{screen['stem']}_valuez.txt")
        lines = open(path, encoding="utf-8", errors="replace").read().splitlines()
        header = lines[0].split("\t")
        rows = [ln.split("\t") for ln in lines[1:] if ln.strip()]
        publication = {"pubmed_id": screen["pmid"], "pubmed_url": ""}
        reference = {"genome_reference": {"background": None}}
        for col in range(1, len(header)):
            if not header[col].strip():
                continue
            meta = _column_meta(header[col])
            if meta is None or "growth" not in (meta["phenotype"] or "").lower():
                continue
            parsed = _parse_conditions(meta["condition"] or "")
            if parsed.unrepresentable:
                continue
            media = _parse_media(meta["media"], meta["phenotype"] or "", screen)
            if prefix and meta["media"] is not None:
                media = media.model_copy(
                    update={
                        "name": meta["media"].split("+")[0].strip(),
                        "components": [],
                        "dropouts": [],
                        "dropout_roles": [],
                    }
                )
            environment = Environment(
                media=media,
                temperature=parsed.temperature,
                perturbations=parsed.perturbations,
                aerobicity=parsed.aerobicity,
                duration_hours=parsed.duration_hours,
                duration_generations=parsed.duration_generations,
                provenance_gaps=(
                    [] if parsed.temperature is not None else [_gap("temperature")]
                ),
            )
            units = f"{NPV_UNITS} [readout: {_readout_method(meta['phenotype'] or '')}]"
            zygosity = meta["zygosity"] or ""
            orfs = [
                cells[0].strip()
                for cells in rows
                if col < len(cells) and cells[col].strip()
            ]
            if not orfs:
                continue
            probe = dataset._experiment(
                orfs[0], 0.0, environment, units, zygosity
            ).model_dump()
            condition = _condition_signature(probe)
            study = _study_key(
                {
                    "experiment": probe,
                    "publication": publication,
                    "reference": reference,
                }
            )
            for orf in orfs:
                experiment = dataset._experiment(
                    orf, 0.0, environment, units, zygosity
                ).model_dump()
                seen.add(
                    (
                        study,
                        _genotype_signature(experiment, frozenset()),
                        None,
                        condition,
                    )
                )
                total += 1
    return {
        "records": total,
        "unique_pair_keys": len(seen),
        "duplicated": total - len(seen),
    }


# --------------------------------------------------------------------------- #
# Is it one dataset's bug? The shared library, and the dev stores
# --------------------------------------------------------------------------- #
def _scan_library() -> dict[str, Any]:
    """The detector over every shared ``MEDIA_LIBRARY`` object (exhaustive)."""
    flagged: dict[str, list[str]] = {}
    for key, media in MEDIA_LIBRARY.items():
        statements = unread_statements(media.model_dump())
        if statements:
            flagged[key] = statements
    return {"media_in_library": len(MEDIA_LIBRARY), "flagged": flagged}


def _interned_table(store: str) -> dict[str, Any]:
    """The store's interned constants, keyed by content hash.

    A store built with interning replaces a constant sub-object (the environment is the
    usual one) with ``{"$ref": <hash>, "name": <hint>}``, so a reader that does not
    splice them back sees a two-key dict and would report every such store as carrying
    no medium at all. ``processed/interned`` is a SEPARATE env, sibling of the records
    env; a store without one reads inline and gets an empty table.
    """
    path = osp.join(store, "processed", "interned")
    if not osp.isdir(path):
        return {}
    env = lmdb.open(path, readonly=True, lock=False, readahead=False, meminit=False)
    try:
        with env.begin() as txn:
            return {key.decode(): pickle.loads(value) for key, value in txn.cursor()}
    finally:
        env.close()


def _resolved(obj: Any, interned: dict[str, Any]) -> Any:
    """``resolve_interned``, duplicated here so the scan reads a store without importing
    the dataset class (which would validate every record under the current closure).
    """
    if isinstance(obj, dict):
        ref = obj.get("$ref")
        if isinstance(ref, str) and ref in interned:
            return _resolved(interned[ref], interned)
        return {k: _resolved(v, interned) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolved(v, interned) for v in obj]
    return obj


def _store_media(
    store: str, cap: int
) -> tuple[list[tuple[dict[str, Any], dict[str, Any]]], int, int]:
    """Distinct stored media of one dev store, sampled on a stride over its keys."""
    interned = _interned_table(store)
    env = lmdb.open(
        osp.join(store, "processed", "lmdb"), readonly=True, lock=False, subdir=True
    )
    try:
        with env.begin() as txn:
            entries = int(txn.stat()["entries"])
            if entries == 0:
                return [], 0, 0
            step = max(1, entries // cap)
            indices = range(0, entries, step)
            seen: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
            read = 0
            for index in indices:
                payload = txn.get(f"{index}".encode())
                if payload is None:
                    continue
                read += 1
                record = pickle.loads(payload)
                experiment = record.get("experiment")
                if not isinstance(experiment, dict):
                    continue
                environment = _resolved(experiment.get("environment"), interned)
                if not isinstance(environment, dict):
                    continue
                media = environment.get("media")
                if not isinstance(media, dict):
                    continue
                seen.setdefault(json.dumps(media, sort_keys=True), (media, environment))
            return list(seen.values()), entries, read
    finally:
        env.close()


def _scan_stores(data_root: str, cap: int) -> dict[str, Any]:
    """The detector over the dev stores, read-only, on a per-store key stride."""
    root = osp.join(data_root, "data", "torchcell")
    out: dict[str, Any] = {}
    unreadable: dict[str, str] = {}
    for name in sorted(os.listdir(root)):
        if "." in name:  # .deprecated / .superseded copies
            continue
        store = osp.join(root, name)
        if not osp.isdir(osp.join(store, "processed", "lmdb")):
            continue
        try:
            media_list, entries, read = _store_media(store, cap)
        except Exception as error:  # a store built under a retired closure
            unreadable[name] = f"{type(error).__name__}: {error}"
            continue
        flagged: dict[str, list[str]] = {}
        for media, environment in media_list:
            statements = unread_statements(media, environment)
            if statements:
                flagged[str(media.get("name"))] = statements
        if flagged:
            out[name] = {
                "records": entries,
                "records_read": read,
                "distinct_media_sampled": len(media_list),
                "flagged": flagged,
            }
    return {"flagged_stores": out, "unreadable_stores": unreadable}


def _scan_growth_phase(data_root: str, cap: int) -> dict[str, Any]:
    """Which stores state ``Environment.growth_phase``, and does reading it split a node?

    The field was in no identity projection at all before lane X of the ontology unification (lane W's finding 2).
    It is projected only when SET, so adding it can only SPLIT: a store that never states
    it keeps every environment id byte-identical, and a store that does keeps them unless
    two of its environments differ ONLY in the phase. This counts both, read-only, on the
    same per-store key stride as the media scan, and reports the number of distinct
    environments whose identity MERGES when the field is dropped from the projection,
    which is the number of records the old projection could have mis-keyed.
    """
    root = osp.join(data_root, "data", "torchcell")
    stating: dict[str, Any] = {}
    unreadable: dict[str, str] = {}
    scanned = 0
    for name in sorted(os.listdir(root)):
        if "." in name:
            continue
        store = osp.join(root, name)
        path = osp.join(store, "processed", "lmdb")
        if not osp.isdir(path):
            continue
        try:
            interned = _interned_table(store)
            env = lmdb.open(path, readonly=True, lock=False, subdir=True)
        except Exception as error:
            unreadable[name] = f"{type(error).__name__}: {error}"
            continue
        try:
            with env.begin() as txn:
                entries = int(txn.stat()["entries"])
                if entries == 0:
                    continue
                scanned += 1
                step = max(1, entries // cap)
                phases: Counter[str] = Counter()
                with_phase: set[str] = set()
                without_phase: set[str] = set()
                read = 0
                for index in range(0, entries, step):
                    payload = txn.get(f"{index}".encode())
                    if payload is None:
                        continue
                    try:
                        record = pickle.loads(payload)
                    except Exception as error:
                        unreadable[name] = f"{type(error).__name__}: {error}"
                        break
                    read += 1
                    experiment = record.get("experiment")
                    if not isinstance(experiment, dict):
                        continue
                    environment = _resolved(experiment.get("environment"), interned)
                    if not isinstance(environment, dict):
                        continue
                    phase = environment.get("growth_phase")
                    phases[str(phase)] += 1
                    with_phase.add(json.dumps(environment, sort_keys=True))
                    stripped = {
                        k: v for k, v in environment.items() if k != "growth_phase"
                    }
                    without_phase.add(json.dumps(stripped, sort_keys=True))
                if any(key != "None" for key in phases):
                    stating[name] = {
                        "records": entries,
                        "records_read": read,
                        "phases": dict(phases.most_common()),
                        "distinct_environments": len(with_phase),
                        "distinct_environments_without_the_phase": len(without_phase),
                        "environments_that_merge_without_it": len(with_phase)
                        - len(without_phase),
                    }
        finally:
            env.close()
    return {
        "stores_scanned": scanned,
        "stores_stating_a_growth_phase": stating,
        "unreadable_stores": unreadable,
    }


def main() -> None:
    """Run the requested passes and write the results JSON."""
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yeastphenome", action="store_true")
    parser.add_argument("--l1", action="store_true")
    parser.add_argument("--library", action="store_true")
    parser.add_argument("--stores", action="store_true")
    parser.add_argument("--growth-phase", action="store_true")
    parser.add_argument(
        "--prefix",
        action="store_true",
        help="emulate the pre-lane X of the ontology unification medium (token truncated at '+', no typed fields)",
    )
    parser.add_argument(
        "--cap",
        type=int,
        default=3000,
        help="records read per dev store, strided over its whole key range",
    )
    parser.add_argument(
        "--out", default=osp.join(RESULTS, "media_name_composition_scan.json")
    )
    args = parser.parse_args()
    data_root = os.environ["DATA_ROOT"]
    raw_dir = osp.join(data_root, "data", "torchcell", "yeastphenome", "raw")

    result: dict[str, Any] = {}
    if args.yeastphenome:
        state = "pre_fix" if args.prefix else "fixed"
        result[f"yeastphenome_tokens_{state}"] = _yeastphenome_rows(
            raw_dir, prefix=args.prefix
        )
        result[f"yeastphenome_environments_{state}"] = _yeastphenome_environments(
            raw_dir, prefix=args.prefix
        )
    if args.l1:
        result["yeastphenome_l1_pre_fix"] = _yeastphenome_l1(raw_dir, prefix=True)
        result["yeastphenome_l1_fixed"] = _yeastphenome_l1(raw_dir, prefix=False)
    if args.library:
        result["media_library"] = _scan_library()
    if args.stores:
        result["dev_stores"] = _scan_stores(data_root, args.cap)
    if args.growth_phase:
        result["growth_phase"] = _scan_growth_phase(data_root, args.cap)

    os.makedirs(RESULTS, exist_ok=True)
    existing: dict[str, Any] = {}
    if osp.exists(args.out):
        existing = json.load(open(args.out))
    existing.update(result)
    with open(args.out, "w") as handle:
        json.dump(existing, handle, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True)[:8000])
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
