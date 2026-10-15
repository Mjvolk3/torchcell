# torchcell/sequence/construct/registry.py
# [[torchcell.sequence.construct.registry]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/sequence/construct/registry.py
# Test file: tests/torchcell/sequence/construct/test_construct_registry.py

"""The construct store: an INDEX over deposited construct sequence, not a fifth tier.

``ArtifactTier`` is ``Literal["raw","genomes","library","objects"]`` and that Literal is
in 120 of 146 loader contract closures, so adding a ``"constructs"`` member would be a
full-rebuild change to buy a directory name. The bytes of a construct therefore live in
the tiers that already exist, by convention:

- a construct the SOURCE released goes to ``library`` under that citation key's own
  ``si/plasmids/`` directory (where the 17 Cachera maps already sit, each sha256-pinned
  in that key's ``manifest.json``);
- a construct retrieved from a REGISTRY (Addgene, the JBEI public registry, an NCBI
  accession for a backbone) goes to ``raw`` under the consuming dataset's key;
- a construct WE reconstruct from parts goes to ``objects``, because it is derived and
  must never be filed as if a source released it.

``$DATA_ROOT/torchcell-constructs/<construct_set>/manifest.json`` holds the index and
NOTHING ELSE: no sequence byte is ever written under it, and every contig reaches its
bytes through an ``ArtifactRef`` into one of the three tiers above, resolved by the one
resolver (``torchcell.artifacts.resolve``), which re-hashes on every read. A second
resolver is what this module deliberately is not.

The shape mirrors ``torchcell.sequence.genome.registry`` member for member so one
reviewer reads both: ``deposit_construct_set`` refuses to overwrite and re-hashes
everything it records, ``extend_construct_set`` is additive and carries every existing
record over byte-identically, and ``resolve_contig`` re-hashes on every call.

Three identity rules, each measured on the 17 mirrored Cachera maps rather than assumed:

1. **A construct's identity is the sha256 of its file**, with the first 12 hex as the
   human handle. NOT the GenBank LOCUS name: ``pBTX002.gb`` carries ``LOCUS pBTX001``
   and ``pBTX001(ante113).gb`` carries ``LOCUS ANTE113_with_NatMX``, so a LOCUS-derived
   id would name the wrong replicon and the paper's ``pBTX1`` would never carry the
   token it was minted from. ``record_name`` keeps the LOCUS line as data.
2. **The source's name is a loader-stated, quoted string** (``source_name``, a
   ``SourcedValue``), or a typed ``ProvenanceGap`` on that field. The replicon's name
   lives in the paper text and the plasmid table, not in the file.
3. **A feature's key is its label, its SO id and its 1-based end-inclusive span.** 0 of
   570 non-source features carry a ``/locus_tag``, so there is no primary key to read;
   label alone collides in 4 of 17 maps and label plus span still collides in 1 (pHO22
   carries a ``gene`` and a ``CDS`` both labelled ``Ampicillin`` at 1543..2403, the
   ordinary GenBank gene/CDS pair), while the triple is unique in all 17.

``ConstructSpan``, ``FeatureKey`` and ``PLACEMENT_TARGET_REFUSED_SO`` are DEFINED in
``torchcell.datamodels.schema`` and re-exported here. They were moved there when
``GeneAdditionPerturbation.placement`` started carrying them (P4): a record class's
field types belong to the fingerprinted schema surface, and this module cannot be the
definition site because it reaches the resolver, which imports the dataset loaders,
which import ``schema``. It is the same split ``torchcell/artifacts/ref.py`` makes for
``ArtifactRef``.

Units: ``plasmid.Location`` is 0-based half-open and ``GenomicSpan`` is 1-based
end-inclusive. A store serving native and construct genes through one extraction path
must not carry two conventions, so the conversion happens here, at the parser boundary,
and a ``ConstructSpan`` is 1-based end-inclusive like every other span in the schema.
"""

from __future__ import annotations

import os
import os.path as osp
from collections.abc import Iterable
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from torchcell.artifacts.ref import ArtifactRef
from torchcell.artifacts.resolve import resolve as resolve_ref
from torchcell.datamodels.pydant import ModelStrict
from torchcell.datamodels.schema import (
    PLACEMENT_TARGET_REFUSED_SO as PLACEMENT_TARGET_REFUSED_SO,
)
from torchcell.datamodels.schema import ConstructSpan as ConstructSpan
from torchcell.datamodels.schema import FeatureKey as FeatureKey
from torchcell.datamodels.schema import SOTerm
from torchcell.literature.manifest import ArtifactRecord, sha256_file
from torchcell.sequence.plasmid import Component, Feature, parse_genbank_component
from torchcell.verification.sourced import ProvenanceGap, SourcedValue

#: The index directory under ``DATA_ROOT``. It holds manifests only, never sequence.
CONSTRUCTS_DIR = "torchcell-constructs"
MANIFEST_FILENAME = "manifest.json"
CONSTRUCT_MANIFEST_VERSION = 1

#: The 17 GenBank maps Cachera et al. 2023 released with the CRI-SPA paper: pBTX1 and
#: pBTX2 (the betaxanthin integration vectors), pCCM023, the pbtx0xx series and the
#: pHO CRISPR helpers, all under that key's ``si/plasmids/`` in the library tier.
CACHERA2023_CRI_SPA_PLASMIDS = "cachera2023_cri_spa_plasmids"


class ConstructIntegrityError(RuntimeError):
    """A construct's bytes, length or feature table disagree with what is recorded."""


class PlacementTargetError(ValueError):
    """A feature was offered as a placement target that may not be one."""


class ConstructKind(StrEnum):
    """What kind of DNA object a contig is, as the source presents it."""

    plasmid = "plasmid"
    integration_vector = "integration_vector"
    cassette = "cassette"
    part = "part"
    reconstruction = "reconstruction"


class SequenceStatus(StrEnum):
    """Whether a deposited sequence is the DESIGN or the sequence of the built DNA.

    A registry (JBEI's ICE, Addgene) serves the design the depositor entered;
    Carruthers 2025 writes "Plasmid sequences were verified by whole plasmid sequencing
    (Primordium Labs)" and deposits nothing. Those are different claims about the same
    named plasmid, so the set says which one its bytes are, and a consumer that needs
    the built molecule can tell. ``design`` is never silently read as ``verified``.
    """

    design = "design"
    verified = "verified"


class BlockedReason(StrEnum):
    """Why a released map is listed by the set but carries no contig."""

    #: Biopython rewrote an origin-wrapping location; the envelope is not the feature.
    origin_wrapping_location = "origin_wrapping_location"
    #: A compound ``join`` location, which a single-range span cannot hold.
    compound_location = "compound_location"


class ConstructFeature(ModelStrict):
    """One projected feature of a contig: its key, its SO roles and its GenBank type.

    ``genbank_type`` is the type line verbatim, kept beside the SO role because the map
    is lossy in one direction that matters: ``misc_feature`` (82 of 570) maps to
    ``engineered_region``, which asserts engineering for a generic feature, and ``mRNA``,
    ``misc_RNA`` and ``oriT`` (15 of 570) fall to ``sequence_feature``. The verbatim
    type is what lets that projection be re-examined without re-reading the bytes.
    """

    key: FeatureKey
    roles: list[SOTerm]
    genbank_type: str

    @model_validator(mode="after")
    def _check_roles(self) -> ConstructFeature:
        """A feature carries at least one role, and its key's id is one of them."""
        if not self.roles:
            raise ValueError("ConstructFeature needs at least one SO role")
        if self.key.so_id not in {role.so_id for role in self.roles}:
            raise ValueError(
                f"feature key names SO id {self.key.so_id} but the feature's roles are "
                f"{[role.so_id for role in self.roles]}"
            )
        return self


class ConstructContig(ModelStrict):
    """One construct sequence, identified by the sha256 of the file that holds it.

    ``source_name`` is the name the SOURCE gives the replicon, as a ``SourcedValue``
    with a verbatim quote, or ``None`` with a ``ProvenanceGap`` naming where we looked.
    It is never derived from the LOCUS line and never from the filename.
    ``record_name`` is the LOCUS line verbatim: data, not a key.
    """

    sha256: str
    ref: ArtifactRef
    record_name: str
    topology: Literal["circular", "linear"]
    length: int
    kind: ConstructKind
    features: list[ConstructFeature]
    source_name: SourcedValue | None = None
    source_statement: str | None = Field(
        default=None,
        description="the release's own one-line description of the construct, verbatim",
    )
    provenance_gaps: list[ProvenanceGap] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_contig(self) -> ConstructContig:
        """Identity is the file's sha256; the name is sourced or gapped; keys are unique."""
        if len(self.sha256) != 64 or any(
            char not in "0123456789abcdef" for char in self.sha256
        ):
            raise ValueError(
                f"ConstructContig.sha256 must be 64 lowercase hex, got {self.sha256!r}"
            )
        if self.ref.sha256 != self.sha256:
            raise ValueError(
                f"contig sha256 {self.sha256} disagrees with its ref's "
                f"{self.ref.sha256}; the identity IS the file's hash"
            )
        if self.length < 1:
            raise ValueError(
                f"ConstructContig.length must be positive, got {self.length}"
            )
        gapped = {gap.field for gap in self.provenance_gaps}
        for gap in self.provenance_gaps:
            if gap.field not in type(self).model_fields:
                raise ValueError(
                    f"provenance_gap field {gap.field!r} is not a field of ConstructContig"
                )
            if getattr(self, gap.field) is not None:
                raise ValueError(
                    f"field {gap.field!r} has a ProvenanceGap but is not None"
                )
        if self.source_name is None and "source_name" not in gapped:
            raise ValueError(
                "ConstructContig.source_name is unset and carries no ProvenanceGap; "
                "the source's name for a construct is quoted or typed absent, never a "
                "silent None and never the LOCUS line"
            )
        tokens = [feature.key.token for feature in self.features]
        if len(set(tokens)) != len(tokens):
            duplicates = sorted({token for token in tokens if tokens.count(token) > 1})
            raise ValueError(
                f"contig {self.handle}: feature keys repeat {duplicates}; the key is "
                "label plus SO id plus span and must be unique within a contig"
            )
        return self

    @property
    def handle(self) -> str:
        """The first 12 hex of the sha256: the readable handle beside the full hash."""
        return self.sha256[:12]

    def feature(self, token: str) -> ConstructFeature:
        """The feature whose key token is ``token``; an unknown token raises ``KeyError``."""
        for candidate in self.features:
            if candidate.key.token == token:
                return candidate
        raise KeyError(f"contig {self.handle} has no feature {token!r}")

    def placement_target(self, token: str) -> ConstructFeature:
        """``feature(token)``, refusing a feature a placement may not point at."""
        found = self.feature(token)
        if not found.key.is_placement_target:
            raise PlacementTargetError(
                f"contig {self.handle} feature {token!r} may not be a placement "
                f"target: {PLACEMENT_TARGET_REFUSED_SO[found.key.so_id]}"
            )
        return found


class BlockedConstruct(ModelStrict):
    """A released file the set lists but could not index, named with its reason.

    A blocked map is REPORTED, never dropped: the file stays in ``files`` with its
    retrieval record and its sha256, and this record says exactly which feature stopped
    it, so the set's coverage is countable and the work to unblock it is named.
    """

    path: str
    reason: BlockedReason
    detail: str = Field(description="the parser's refusal, verbatim")


class ConstructManifest(BaseModel):
    """Provenance and integrity record for one construct set (the index).

    ``files`` reuses the literature ``ArtifactRecord`` exactly as ``GenomeManifest``
    does, so a reviewer reads one record shape everywhere. ``provenance_complete`` keeps
    the literature meaning and is about RETRIEVAL only: True when every listed file
    carries its own ``RetrievalRecord``, False when bytes are pinned but the retrieval
    chain is not reproducible. It says nothing about whether a construct's name is
    sourced (that is each contig's ``source_name``) or whether every map parsed (that is
    ``blocked``); conflating the three is how a manifest comes to claim more than it has.
    """

    model_config = ConfigDict(extra="forbid")

    version: int = Field(default=CONSTRUCT_MANIFEST_VERSION)
    construct_set: str = Field(
        description="Directory name under torchcell-constructs/."
    )
    source: str = Field(description='"Cachera et al. 2023 supplementary", "Addgene".')
    sequence_status: SequenceStatus
    citation_key: str | None = None
    files: list[ArtifactRecord] = Field(default_factory=list)
    contigs: list[ConstructContig] = Field(default_factory=list)
    blocked: list[BlockedConstruct] = Field(default_factory=list)
    provenance_complete: bool
    created_at: str
    notes: str | None = None

    @model_validator(mode="after")
    def _check_manifest(self) -> ConstructManifest:
        """Contig handles are unique, every contig's file is listed, completeness is earned."""
        handles = [contig.handle for contig in self.contigs]
        if len(set(handles)) != len(handles):
            raise ValueError(
                f"{self.construct_set}: contig handles repeat in {handles}"
            )
        listed = {record.path: record for record in self.files}
        for contig in self.contigs:
            record = listed.get(contig.ref.path)
            if record is None:
                raise ValueError(
                    f"{self.construct_set}: contig {contig.handle} points at "
                    f"{contig.ref.path!r}, which the manifest does not list"
                )
            if record.sha256 != contig.sha256:
                raise ValueError(
                    f"{self.construct_set}: contig {contig.handle} pins "
                    f"{contig.sha256} but the file record has {record.sha256}"
                )
        for entry in self.blocked:
            if entry.path not in listed:
                raise ValueError(
                    f"{self.construct_set}: blocked {entry.path!r} is not listed"
                )
        complete = all(record.retrieval is not None for record in self.files)
        if self.provenance_complete and not complete:
            missing = sorted(
                record.path for record in self.files if record.retrieval is None
            )
            raise ValueError(
                f"{self.construct_set}: provenance_complete is True but "
                f"{missing} carry no RetrievalRecord; completeness is per file"
            )
        return self

    def contig(self, handle: str) -> ConstructContig:
        """The contig with this handle (or full sha256); an unknown one raises ``KeyError``."""
        for candidate in self.contigs:
            if handle in (candidate.handle, candidate.sha256):
                return candidate
        raise KeyError(f"{self.construct_set}: no contig {handle!r}")

    def record(self, path: str) -> ArtifactRecord:
        """The file record for ``path``; a path the manifest does not list raises."""
        for candidate in self.files:
            if candidate.path == path:
                return candidate
        raise KeyError(f"{self.construct_set}: {path!r} is not in the manifest")


def constructs_root(data_root: str | None = None) -> str:
    """``$DATA_ROOT/torchcell-constructs``; ``DATA_ROOT`` comes from the environment."""
    if data_root is None:
        from dotenv import load_dotenv

        load_dotenv()
        data_root = os.environ["DATA_ROOT"]
    return osp.join(data_root, CONSTRUCTS_DIR)


def construct_set_dir(construct_set: str, data_root: str | None = None) -> str:
    """Absolute directory of one construct set's index (not checked for existence)."""
    return osp.join(constructs_root(data_root), construct_set)


def load_construct_manifest(
    construct_set: str, data_root: str | None = None
) -> ConstructManifest:
    """Validate and return ``<constructs root>/<construct_set>/manifest.json``."""
    path = osp.join(construct_set_dir(construct_set, data_root), MANIFEST_FILENAME)
    if not osp.isfile(path):
        raise FileNotFoundError(
            f"construct set {construct_set!r} has no index at {path}; deposit it with "
            "scripts/provision_construct_sets.py"
        )
    manifest = ConstructManifest.model_validate_json(
        Path(path).read_text(encoding="utf-8")
    )
    if manifest.construct_set != construct_set:
        raise ConstructIntegrityError(
            f"{path} names construct set {manifest.construct_set!r}, "
            f"not {construct_set!r}"
        )
    return manifest


def resolve_contig(
    construct_set: str, handle: str, *, data_root: str | None = None
) -> Path:
    """The verified local path of one contig's bytes, re-hashed on every call.

    Delegates to the ONE resolver (``torchcell.artifacts.resolve``), which checks the
    tier manifest, then the bytes, and raises ``ArtifactIntegrityError`` on any sha256
    disagreement. This function adds the construct set's own claim to that chain: the
    resolved bytes must also hash to the sha256 the contig IS.
    """
    contig = load_construct_manifest(construct_set, data_root).contig(handle)
    resolved = resolve_ref(contig.ref, materialize=True, data_root=data_root)
    got = sha256_file(resolved.path)
    if got != contig.sha256:
        raise ConstructIntegrityError(
            f"{construct_set}/{contig.handle}: sha256 {got} on disk, the contig IS "
            f"{contig.sha256}"
        )
    return resolved.path


def parse_contig_features(
    path: str | Path, citation_key: str
) -> tuple[Component, list[ConstructFeature]]:
    """Parse a GenBank map and project its features into 1-based end-inclusive keys.

    Raises ``GenBankLocationError`` (from the parser) for an origin-wrapping or compound
    location, which is how a map becomes a ``BlockedConstruct`` instead of a contig.
    """
    component = parse_genbank_component(str(path), citation_key)
    return component, [_project(feature) for feature in component.features]


def _project(feature: Feature) -> ConstructFeature:
    """One ``plasmid.Feature`` (0-based half-open) as a ``ConstructFeature`` (1-based)."""
    roles = list(feature.roles)
    return ConstructFeature(
        key=FeatureKey(
            label=feature.name,
            so_id=roles[0].so_id,
            span=ConstructSpan(
                start=feature.location.start + 1,
                end=feature.location.end,
                orientation=feature.location.orientation,
            ),
        ),
        roles=roles,
        genbank_type=_GENBANK_TYPE_BY_SO.get(roles[0].so_id, roles[0].name),
    )


#: The GenBank type a projected SO role came from, for the roles the parser maps. A role
#: several types share (``primer`` and ``primer_bind`` both mean a primer binding site)
#: reports the SO name, since the projection cannot tell them apart.
_GENBANK_TYPE_BY_SO: dict[str, str] = {
    "SO:0000704": "gene",
    "SO:0000316": "CDS",
    "SO:0000167": "promoter",
    "SO:0000141": "terminator",
    "SO:0000296": "rep_origin",
    "SO:0000139": "RBS",
    "SO:0000804": "misc_feature",
    "SO:0000410": "protein_bind",
}


def verify_construct_set(
    construct_set: str, data_root: str | None = None
) -> dict[str, str]:
    """Re-resolve and re-hash every contig; return ``{handle: sha256}``; raise on a defect.

    Also re-measures each contig's length from the bytes, because a length copied from a
    GenBank header is a claim and the bytes are the fact.
    """
    manifest = load_construct_manifest(construct_set, data_root)
    measured: dict[str, str] = {}
    for contig in manifest.contigs:
        path = resolve_contig(construct_set, contig.handle, data_root=data_root)
        component = parse_genbank_component(str(path), manifest.citation_key or "")
        if component.length != contig.length:
            raise ConstructIntegrityError(
                f"{construct_set}/{contig.handle}: the bytes are {component.length} bp, "
                f"the index records {contig.length}"
            )
        measured[contig.handle] = contig.sha256
    return measured


def deposit_construct_set(
    manifest: ConstructManifest, data_root: str | None = None
) -> str:
    """Write ``manifest.json`` for a construct set whose bytes are already in a tier.

    Refuses to overwrite an existing index, refuses an empty set, and refuses a manifest
    whose records disagree with the bytes the tiers hold: every listed file is resolved
    through the one resolver and re-hashed, and every contig's length is re-measured
    from its bytes. An index can therefore never claim a hash or a length it did not
    measure. Returns the manifest path.
    """
    directory = construct_set_dir(manifest.construct_set, data_root)
    path = osp.join(directory, MANIFEST_FILENAME)
    if osp.exists(path):
        raise FileExistsError(f"{path} exists; a deposited index is never rewritten")
    if not manifest.files:
        raise ValueError(f"{manifest.construct_set}: an index must list its files")
    if not manifest.contigs and not manifest.blocked:
        raise ValueError(
            f"{manifest.construct_set}: an index with no contig and nothing blocked "
            "records nothing"
        )
    for contig in manifest.contigs:
        resolved = resolve_ref(contig.ref, materialize=True, data_root=data_root)
        got = sha256_file(resolved.path)
        if got != contig.sha256:
            raise ConstructIntegrityError(
                f"{manifest.construct_set}/{contig.handle}: the index pins "
                f"{contig.sha256}, {resolved.path} has {got}"
            )
        component = parse_genbank_component(
            str(resolved.path), manifest.citation_key or ""
        )
        if component.length != contig.length:
            raise ConstructIntegrityError(
                f"{manifest.construct_set}/{contig.handle}: the bytes are "
                f"{component.length} bp, the index says {contig.length}"
            )
        if component.topology != contig.topology:
            raise ConstructIntegrityError(
                f"{manifest.construct_set}/{contig.handle}: the bytes say "
                f"{component.topology!r}, the index says {contig.topology!r}"
            )
    Path(directory).mkdir(parents=True, exist_ok=True)
    Path(path).write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return path


def extend_construct_set(
    construct_set: str,
    new_files: Iterable[ArtifactRecord],
    new_contigs: Iterable[ConstructContig] = (),
    new_blocked: Iterable[BlockedConstruct] = (),
    data_root: str | None = None,
) -> str:
    """Append files, contigs and blocked entries to a deposited index; return its path.

    ``deposit_construct_set`` never rewrites an index, because rewriting could change a
    hash a stored record already points at. ADDING changes no existing claim, so this is
    the one sanctioned rewrite, exactly as ``extend_assembly_set`` is for the genomes
    tier: every existing record is carried over unchanged, a path or a handle the index
    already holds is refused, and each new contig is resolved and re-hashed as a first
    deposit checks it. ``provenance_complete`` is recomputed, because a file with no
    retrieval record arriving makes a complete set incomplete.
    """
    files = list(new_files)
    contigs = list(new_contigs)
    blocked = list(new_blocked)
    if not files and not contigs and not blocked:
        raise ValueError(f"{construct_set}: nothing to add")
    manifest = load_construct_manifest(construct_set, data_root)
    listed = {record.path for record in manifest.files}
    clashes = sorted(record.path for record in files if record.path in listed)
    if clashes:
        raise FileExistsError(
            f"{construct_set}: {clashes} already in the index; a listed file is never "
            "re-recorded"
        )
    held = {contig.handle for contig in manifest.contigs}
    repeats = sorted(contig.handle for contig in contigs if contig.handle in held)
    if repeats:
        raise FileExistsError(f"{construct_set}: contigs {repeats} are already indexed")
    for contig in contigs:
        resolved = resolve_ref(contig.ref, materialize=True, data_root=data_root)
        got = sha256_file(resolved.path)
        if got != contig.sha256:
            raise ConstructIntegrityError(
                f"{construct_set}/{contig.handle}: the index pins {contig.sha256}, "
                f"{resolved.path} has {got}"
            )
    all_files = [*manifest.files, *files]
    extended = manifest.model_copy(
        update={
            "files": all_files,
            "contigs": [*manifest.contigs, *contigs],
            "blocked": [*manifest.blocked, *blocked],
            "provenance_complete": all(
                record.retrieval is not None for record in all_files
            ),
        }
    )
    ConstructManifest.model_validate(extended.model_dump())
    path = osp.join(construct_set_dir(construct_set, data_root), MANIFEST_FILENAME)
    Path(path).write_text(extended.model_dump_json(indent=2), encoding="utf-8")
    return path


def utc_now() -> str:
    """ISO-8601 UTC timestamp, for a manifest's ``created_at``."""
    return datetime.now(UTC).isoformat()
