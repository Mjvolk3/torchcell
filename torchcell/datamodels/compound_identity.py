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
  (``compound_identity_curate.py`` is the only thing that hits PubChem, run by a
  human against the committed input lists in ``compound_identity_inputs/``). The
  bytes are sha256-self-checked at import; a mismatch raises (tamper / drift
  detection).
- **Non-generic pydantic.** ``CompoundIdentityRecord`` /
  ``CompoundIdentityResolution`` are plain ``BaseModel`` (no ``Generic[...]``) so
  loaders embedding them stay pickle/multiprocessing-safe (PR #119 lesson).
- **Callers own retention.** The resolver never mutates or drops -- it REPORTS a
  typed status. :func:`resolved_compound` is a thin convenience that a loader uses
  to build a ``Compound`` fill-or-gap: fill structure fields where resolved,
  attach a typed ``ProvenanceGap`` on ``inchikey`` where not -- **additive only**,
  never clobbering a caller-supplied ``smiles`` (hoepfner already sets it).

Canonical-name policy (serve-50)
--------------------------------
A row's ``name`` is the ONE name a resolved compound carries, and
:func:`resolved_compound` returns it in ``Compound.name`` rather than echoing the
label the loader passed. Every other spelling a paper uses is a ``synonyms`` entry on
that row, so a loader passes its SOURCE LABEL as a lookup key and gets the canonical
identity back. A row's name is PubChem's ``Title`` lowercased unless an input line
curated a spelling (``media.py`` owns ``D-glucose``; the pre-existing table owns
``actinomycin D``). This is what collapses ``NaCl`` and ``sodium chloride`` onto one
``environment perturbation`` node instead of two joinable only on the ``inchikey``
property, and it is why the source label must NOT be written into ``Compound.name``.
``Compound`` has no ``source_label`` slot and ``schema.py`` is not changed to add one:
``Compound`` sits in every served dataset's closure, so a field there is a full-rebuild
trigger. The label survives in the table's ``synonyms``, which is where it belongs.

Two structure routes, and the curated one wins
----------------------------------------------
:func:`inchikey_from_smiles` derives an InChIKey from a released SMILES with RDKit, for
datasets that publish structures but no names a table can key on (Hoepfner 2014: 151 of
152 SMILES parse, 150 distinct keys). ``resolved_compound(..., derive_from_smiles=True)``
uses it ONLY when the table has no row for the label, because the two routes disagree:
concanamycin A's curated PubChem key and its SMILES-derived key differ in the stereo
block. A curated row therefore always wins, and the derivation route is reported as
``RESOLVED_FROM_SMILES``.
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
_TABLE_SHA256 = "da91da75e84743e4b782600ead0fe3bc220c4bdc02ee8f6d89734de4934c037c"

# Conservative, DOCUMENTED synonym canonicalization (normalized -> normalized),
# applied BEFORE lookup. Only spellings we are certain name the SAME compound --
# never a fuzzy near-miss.
#
# Most aliases now live in the TABLE (each row's ``synonyms``, sourced from the
# committed input lists), which is the reproducible home for them. This code-level
# map is kept for the handful of foldings that predate the table and for spellings
# no input list claims; the two compose, since normalization runs first and the
# resulting key is then looked up among names AND synonyms.
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
    """Typed outcome of a resolution attempt, ordered from usable to terminal.

    - ``RESOLVED``: the table maps the name/CID to a structure (an InChIKey).
    - ``RESOLVED_FROM_SMILES``: no table row, but the caller's released SMILES
      parsed and yielded an InChIKey (the Hoepfner route).
    - ``RESOLVED_MIXTURE``: identified as a substance by ChEBI and/or a PubChem CID,
      but no single-molecule InChIKey exists. Tunicamycin is at least ten homologues
      (ChEBI:29699); agar's PubChem InChIKey belongs to the agarobiose repeat unit,
      not to the algal polysaccharide. The identity rule accepts ChEBI or CID here,
      and fabricating a key would misstate what was in the flask.
    - ``UNRESOLVED_PUBLIC``: a real public name we simply have not resolved yet --
      the ONE recoverable gap (grows the table over time).
    - ``UNDEFINED_MIXTURE``: an intrinsically undefined preparation (yeast extract,
      peptone, Difco YNB, an SC dropout powder). No structure exists to find --
      terminal, and NOT a worklist item.
    - ``PROPRIETARY``: a vendor or catalog code (Novartis CMBxxx, a ChemDiv or
      ChemBridge library id) whose structure the primary never released -- terminal.
    """

    RESOLVED = "RESOLVED"
    RESOLVED_FROM_SMILES = "RESOLVED_FROM_SMILES"
    RESOLVED_MIXTURE = "RESOLVED_MIXTURE"
    UNRESOLVED_PUBLIC = "UNRESOLVED_PUBLIC"
    UNDEFINED_MIXTURE = "UNDEFINED_MIXTURE"
    PROPRIETARY = "PROPRIETARY"


#: Statuses whose gap on ``inchikey`` is TERMINAL: there is nothing left to fetch.
_TERMINAL_STATUSES = frozenset(
    {
        CompoundResolutionStatus.RESOLVED_MIXTURE,
        CompoundResolutionStatus.UNDEFINED_MIXTURE,
        CompoundResolutionStatus.PROPRIETARY,
    }
)


class CompoundIdentityRecord(BaseModel):
    """One row of the pinned name->structure table (JSON round-trips this natively)."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="the ONE canonical name this compound carries")
    inchikey: str | None = Field(default=None, description="canonical InChIKey")
    pubchem_cid: int | None = Field(default=None, description="PubChem CID integer")
    chebi_id: str | None = Field(default=None, description="ChEBI CURIE")
    smiles: str | None = Field(
        default=None, description="canonical/connectivity SMILES"
    )
    synonyms: list[str] = Field(
        default_factory=list,
        description="every other label a source spells this compound with; each is a "
        "lookup key onto this row",
    )
    source_url: str | None = Field(
        default=None, description="the exact retrieval endpoint queried"
    )
    retrieval_method: str | None = Field(
        default=None, description="RetrievalMethod value, e.g. 'pubchem_api'"
    )
    retrieved_at: str | None = Field(
        default=None, description="ISO date the curator queried PubChem"
    )
    resolution_status: str = Field(
        description="CompoundResolutionStatus value at curation time"
    )
    unresolved_reason: str | None = Field(
        default=None,
        description="why this row carries no InChIKey; the audit trail behind a drop",
    )


class CompoundIdentityResolution(BaseModel):
    """What a caller splats into ``Compound`` (only the identity-carrying fields)."""

    model_config = ConfigDict(extra="forbid")

    status: CompoundResolutionStatus
    name: str | None = Field(
        default=None, description="the table's canonical name; None when no row matched"
    )
    inchikey: str | None = None
    chebi_id: str | None = None
    pubchem_cid: int | None = None
    smiles: str | None = None
    unresolved_reason: str | None = Field(
        default=None, description="the row's audit trail for an absent InChIKey"
    )

    @property
    def curated(self) -> bool:
        """Whether a table row matched at all (the SMILES route defers to this)."""
        return self.name is not None

    @property
    def identified(self) -> bool:
        """Whether ANY structure identifier is carried (InChIKey, ChEBI, or CID).

        This is the retention rule the brief states: a compound with none of these
        cannot be encoded, so records depending on it are dropped rather than served
        under a bare name.
        """
        return (
            self.inchikey is not None
            or self.chebi_id is not None
            or self.pubchem_cid is not None
        )


def normalize_compound_name(name: str) -> str:
    """Lowercase + strip + conservative synonym canonicalization (documented)."""
    key = name.strip().lower()
    return _SYNONYMS.get(key, key)


def _load_table() -> tuple[
    list[CompoundIdentityRecord],
    dict[str, CompoundIdentityRecord],
    dict[int, CompoundIdentityRecord],
]:
    """Read + sha256-self-check the committed table, index by name, synonym, and CID.

    Every lookup key must point at exactly ONE compound. A key claimed by two rows is
    raised, not silently won by whichever row loaded last: the curator already awards a
    contested key by precedence and qualifies the losers, so a duplicate here means the
    table and this module's ``_SYNONYMS`` folding disagree and a loader would otherwise
    get a different compound than the one it asked for.
    """
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
        for label in [record.name, *record.synonyms]:
            key = normalize_compound_name(label)
            held = by_name.get(key)
            if held is not None and held is not record:
                raise RuntimeError(
                    f"compound_identity_table.json: lookup key {key!r} is claimed by "
                    f"both {held.name!r} and {record.name!r}"
                )
            by_name[key] = record
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

    Looks up by normalized ``name`` (matching a row's canonical name OR any of its
    synonyms) first, then by ``pubchem_cid``. A matched row's own
    ``resolution_status`` is authoritative, so a mixture comes back as
    ``RESOLVED_MIXTURE`` carrying ChEBI/CID and no InChIKey, and a vendor code comes
    back ``PROPRIETARY`` with the reason it is terminal. With no row at all the answer
    is ``PROPRIETARY`` when the caller knows the label is a vendor code, else
    ``UNRESOLVED_PUBLIC``. NEVER guesses a structure from a near-miss name.
    """
    record: CompoundIdentityRecord | None = None
    if name is not None:
        record = _BY_NAME.get(normalize_compound_name(name))
    if record is None and pubchem_cid is not None:
        record = _BY_CID.get(pubchem_cid)

    if record is None:
        return CompoundIdentityResolution(
            status=(
                CompoundResolutionStatus.PROPRIETARY
                if known_proprietary
                else CompoundResolutionStatus.UNRESOLVED_PUBLIC
            )
        )

    return CompoundIdentityResolution(
        status=CompoundResolutionStatus(record.resolution_status),
        name=record.name,
        inchikey=record.inchikey,
        chebi_id=record.chebi_id,
        pubchem_cid=record.pubchem_cid,
        smiles=record.smiles,
        unresolved_reason=record.unresolved_reason,
    )


def inchikey_from_smiles(smiles: str) -> str | None:
    """Derive a standard InChIKey from a SMILES with RDKit, or ``None`` if it will not parse.

    The offline structure route for a dataset that releases SMILES but no name a curated
    table can key on. ``None`` is a MEASUREMENT, not a swallowed error: RDKit's parser
    returns no molecule for chemistry it cannot represent (Hoepfner's boromycin, whose
    boron cage fails), and it logs the reason to stderr. RDKit is imported lazily so this
    module stays cheap to import in a loader worker.
    """
    from rdkit import Chem
    from rdkit.Chem import inchi

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    key: str = inchi.MolToInchiKey(molecule)  # type: ignore[no-untyped-call]  # RDKit untyped
    return key or None


def resolve_compound_identity_from_smiles(
    name: str, smiles: str
) -> CompoundIdentityResolution:
    """Resolve by name, falling back to the caller's SMILES ONLY when no row exists.

    The curated row always wins: the two routes measurably disagree (concanamycin A's
    curated PubChem key and its SMILES-derived key differ in the stereo block), so a
    dataset that both appears in the table and ships a SMILES must not silently get a
    second identity for the same compound.
    """
    resolution = resolve_compound_identity(name=name)
    if resolution.curated:
        return resolution
    derived = inchikey_from_smiles(smiles)
    if derived is None:
        return resolution
    return CompoundIdentityResolution(
        status=CompoundResolutionStatus.RESOLVED_FROM_SMILES,
        inchikey=derived,
        smiles=smiles,
    )


def resolved_compound(
    name: str,
    *,
    pubchem_cid: int | None = None,
    smiles: str | None = None,
    inchi: str | None = None,
    roles: list[str] | None = None,
    known_proprietary: bool = False,
    derive_from_smiles: bool = False,
) -> Compound:
    """Build a ``Compound`` fill-or-gap through the resolver (the loader entrypoint).

    ``name`` is the loader's SOURCE LABEL: whatever the paper's column or table calls
    the compound. The returned ``Compound.name`` is the table's CANONICAL name when a
    row matches, so two papers spelling one compound differently produce one node; the
    label itself is already recorded as that row's synonym. With no row, the label is
    kept as-is.

    Structure fields merge ADDITIVELY: a caller-supplied field (hoepfner's ``smiles``,
    wildenhain's ``pubchem_cid``) always wins; the resolver only fills what is still
    ``None``. With ``derive_from_smiles=True`` an unmatched label whose SMILES parses
    gets an RDKit-derived InChIKey, but a curated row always outranks that route.

    When no InChIKey can be filled, a typed ``ProvenanceGap`` is attached on
    ``inchikey``: ``not_reported_by_primary`` when the absence is terminal (a vendor
    code, an undefined preparation, a homolog mixture with no single key), else
    ``deferred_pending_source_review`` (the growable worklist). The gap is asserted ONLY
    on the ``None`` field, honoring the ``ProvenanceGapMixin`` invariant.
    """
    resolution = resolve_compound_identity(
        name=name, pubchem_cid=pubchem_cid, known_proprietary=known_proprietary
    )
    if (
        derive_from_smiles
        and smiles is not None
        and resolution.inchikey is None
        and not resolution.curated
    ):
        derived = inchikey_from_smiles(smiles)
        if derived is not None:
            resolution = resolution.model_copy(
                update={
                    "status": CompoundResolutionStatus.RESOLVED_FROM_SMILES,
                    "inchikey": derived,
                }
            )

    merged_inchikey = resolution.inchikey  # loaders never supply an inchikey
    merged_cid = pubchem_cid if pubchem_cid is not None else resolution.pubchem_cid
    merged_smiles = smiles if smiles is not None else resolution.smiles

    gaps: list[ProvenanceGap] = []
    if merged_inchikey is None:
        reason = (
            ProvenanceGapReason.not_reported_by_primary
            if resolution.status in _TERMINAL_STATUSES
            else ProvenanceGapReason.deferred_pending_source_review
        )
        gaps.append(ProvenanceGap(field="inchikey", reason=reason))

    return Compound(
        name=resolution.name if resolution.name is not None else name,
        inchikey=merged_inchikey,
        inchi=inchi,
        smiles=merged_smiles,
        pubchem_cid=merged_cid,
        chebi_id=resolution.chebi_id,
        roles=roles or [],
        provenance_gaps=gaps,
    )
