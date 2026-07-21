# torchcell/datamodels/compound_identity
# [[torchcell.datamodels.compound_identity]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datamodels/compound_identity
"""Shared, pure, offline compound-identity resolver (UI-2).

Every environmental dataset used to store a compound *name only* -- a
``Compound(name="furfural")`` with ``inchikey``/``chebi_id``/``pubchem_cid``/
``smiles`` all ``None``. A bare name is not a resolvable identity: two papers
spelling "H2O2" vs "hydrogen peroxide" become distinct compounds and nothing
joins to ChEBI/PubChem. This module is the ONE reconciler that turns a name (or a
PubChem CID) into a structure -- OR reports an honest, typed absence.

Design (mirrors ``SCerevisiaeGenome.resolve_gene_name``):

- **Pure + offline.** :func:`resolve_compound_identity` reads ONLY the committed
  ``compound_identity_table.json`` -- it NEVER touches the network at import,
  build, CI, or test time. The table is the sha256-pinned canonical artifact
  (``scripts/build_compound_identity_table.py`` is the only thing that hits
  PubChem, run once by a human). The bytes are sha256-self-checked at import; a
  mismatch raises (tamper / drift detection).
- **Non-generic pydantic.** ``CompoundIdentityRecord`` /
  ``CompoundIdentityResolution`` are plain ``BaseModel`` (no ``Generic[...]``) so
  loaders embedding them stay pickle/multiprocessing-safe (PR #119 lesson).
- **Callers own retention.** The resolver never mutates or drops -- it REPORTS a
  typed status. :func:`resolved_compound` is a thin convenience that a loader uses
  to build a ``Compound`` fill-or-gap: fill structure fields where resolved,
  attach a typed ``ProvenanceGap`` on ``inchikey`` where not -- **additive only**,
  never clobbering a caller-supplied ``smiles`` (hoepfner already sets it).
"""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from torchcell.datamodels.schema import Compound
from torchcell.verification.sourced import ProvenanceGap, ProvenanceGapReason

# The committed table lives next to this module; its bytes are the canonical,
# sha256-pinned artifact. The builder recomputes + prints this constant.
_TABLE_PATH = Path(__file__).with_name("compound_identity_table.json")
_TABLE_SHA256 = "9e42cd290c5dadd858d2d9e39dcbcbd7d8dd945e251c1b6b73998cd1f0853f79"

# Conservative, DOCUMENTED synonym canonicalization (normalized -> normalized).
# Only spellings we are certain name the SAME compound -- never a fuzzy near-miss.
# Loaders may spell a compound differently from the seeded canonical name; this
# maps those aliases onto the canonical key BEFORE lookup.
_SYNONYMS: dict[str, str] = {
    "h2o2": "hydrogen peroxide",
    "hydrogen peroxide (h2o2)": "hydrogen peroxide",
    "nacl": "sodium chloride",
    "sodium chloride (nacl)": "sodium chloride",
    "mms": "methyl methanesulfonate",
    "methylmethane sulfonate": "methyl methanesulfonate",
    "actinomycin-d": "actinomycin d",
    "concanamycin-a": "concanamycin a",
    "acetate": "acetic acid",
    "1-propanol": "1-propanol",
    "n-propanol": "1-propanol",
    "propan-1-ol": "1-propanol",
    "etoh": "ethanol",
}


class CompoundResolutionStatus(StrEnum):
    """Typed outcome of a resolution attempt.

    - ``RESOLVED``: the table maps the name/CID to a structure (an InChIKey).
    - ``UNRESOLVED_PUBLIC``: a real public name we simply have not resolved yet --
      the ONE recoverable gap (grows the table over time).
    - ``PROPRIETARY``: a known-proprietary code (e.g. Novartis CMBxxx) whose
      structure the primary never released -- terminal.
    """

    RESOLVED = "RESOLVED"
    UNRESOLVED_PUBLIC = "UNRESOLVED_PUBLIC"
    PROPRIETARY = "PROPRIETARY"


class CompoundIdentityRecord(BaseModel):
    """One row of the pinned name->structure table (JSON round-trips this natively)."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="canonical human name, or 'CID <cid>' for a CID row")
    inchikey: str | None = Field(default=None, description="canonical InChIKey")
    pubchem_cid: int | None = Field(default=None, description="PubChem CID integer")
    chebi_id: str | None = Field(default=None, description="ChEBI CURIE (hand-curated)")
    smiles: str | None = Field(
        default=None, description="canonical/connectivity SMILES"
    )
    source_url: str | None = Field(
        default=None, description="the exact retrieval endpoint queried"
    )
    retrieval_method: str | None = Field(
        default=None, description="RetrievalMethod value, e.g. 'pubchem_api'"
    )
    retrieved_at: str | None = Field(
        default=None, description="ISO date the builder queried PubChem"
    )
    resolution_status: str = Field(
        description="CompoundResolutionStatus value at build time"
    )


class CompoundIdentityResolution(BaseModel):
    """What a caller splats into ``Compound`` (only the identity-carrying fields)."""

    model_config = ConfigDict(extra="forbid")

    status: CompoundResolutionStatus
    inchikey: str | None = None
    chebi_id: str | None = None
    pubchem_cid: int | None = None
    smiles: str | None = None


def normalize_compound_name(name: str) -> str:
    """Lowercase + strip + conservative synonym canonicalization (documented)."""
    key = name.strip().lower()
    return _SYNONYMS.get(key, key)


def _load_table() -> tuple[
    list[CompoundIdentityRecord],
    dict[str, CompoundIdentityRecord],
    dict[int, CompoundIdentityRecord],
]:
    """Read + sha256-self-check the committed table, index by normalized name + CID."""
    raw = _TABLE_PATH.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != _TABLE_SHA256:
        raise RuntimeError(
            f"compound_identity_table.json sha256 mismatch: got {digest}, expected "
            f"{_TABLE_SHA256} (table tampered or re-built -- re-pin _TABLE_SHA256)"
        )
    payload = json.loads(raw)
    records = [CompoundIdentityRecord.model_validate(r) for r in payload["records"]]
    by_name: dict[str, CompoundIdentityRecord] = {}
    by_cid: dict[int, CompoundIdentityRecord] = {}
    for record in records:
        by_name[normalize_compound_name(record.name)] = record
        if record.pubchem_cid is not None:
            by_cid.setdefault(record.pubchem_cid, record)
    return records, by_name, by_cid


_RECORDS, _BY_NAME, _BY_CID = _load_table()


def resolve_compound_identity(
    name: str | None = None,
    pubchem_cid: int | None = None,
    known_proprietary: bool = False,
) -> CompoundIdentityResolution:
    """Resolve a compound identity from the pinned table -- pure, offline.

    Looks up by normalized ``name`` first, then by ``pubchem_cid``. Returns
    ``RESOLVED`` with the structure fields when the table carries an InChIKey; else
    ``PROPRIETARY`` (when ``known_proprietary``) or ``UNRESOLVED_PUBLIC``. NEVER
    guesses a structure from a near-miss name.
    """
    record: CompoundIdentityRecord | None = None
    if name is not None:
        record = _BY_NAME.get(normalize_compound_name(name))
    if record is None and pubchem_cid is not None:
        record = _BY_CID.get(pubchem_cid)

    if record is not None and record.inchikey is not None:
        return CompoundIdentityResolution(
            status=CompoundResolutionStatus.RESOLVED,
            inchikey=record.inchikey,
            chebi_id=record.chebi_id,
            pubchem_cid=record.pubchem_cid,
            smiles=record.smiles,
        )

    status = (
        CompoundResolutionStatus.PROPRIETARY
        if known_proprietary
        else CompoundResolutionStatus.UNRESOLVED_PUBLIC
    )
    # A row may exist but be unresolved (PubChem failed at build time); surface any
    # cross-reference it does carry, but leave inchikey None so the caller gaps it.
    return CompoundIdentityResolution(
        status=status,
        pubchem_cid=record.pubchem_cid if record is not None else None,
        smiles=record.smiles if record is not None else None,
    )


def resolved_compound(
    name: str,
    *,
    pubchem_cid: int | None = None,
    smiles: str | None = None,
    inchi: str | None = None,
    roles: list[str] | None = None,
    known_proprietary: bool = False,
) -> Compound:
    """Build a ``Compound`` fill-or-gap through the resolver (the loader entrypoint).

    Merges resolver output ADDITIVELY: a caller-supplied structure field (e.g.
    hoepfner's ``smiles``, wildenhain's ``pubchem_cid``) always wins; the resolver
    only fills a field that is still ``None``. When no InChIKey can be filled, a
    typed ``ProvenanceGap`` is attached on ``inchikey`` -- ``not_reported_by_primary``
    for a known-proprietary code, else ``deferred_pending_source_review`` (the
    growable worklist). The gap is asserted ONLY on the ``None`` field, honoring the
    ``ProvenanceGapMixin`` invariant.
    """
    resolution = resolve_compound_identity(
        name=name, pubchem_cid=pubchem_cid, known_proprietary=known_proprietary
    )
    merged_inchikey = resolution.inchikey  # loaders never supply an inchikey
    merged_cid = pubchem_cid if pubchem_cid is not None else resolution.pubchem_cid
    merged_smiles = smiles if smiles is not None else resolution.smiles
    merged_chebi = resolution.chebi_id

    gaps: list[ProvenanceGap] = []
    if merged_inchikey is None:
        reason = (
            ProvenanceGapReason.not_reported_by_primary
            if resolution.status == CompoundResolutionStatus.PROPRIETARY
            else ProvenanceGapReason.deferred_pending_source_review
        )
        gaps.append(ProvenanceGap(field="inchikey", reason=reason))

    return Compound(
        name=name,
        inchikey=merged_inchikey,
        inchi=inchi,
        smiles=merged_smiles,
        pubchem_cid=merged_cid,
        chebi_id=merged_chebi,
        roles=roles or [],
        provenance_gaps=gaps,
    )
