# tests/torchcell/sequence/construct/test_construct_registry.py
# [[tests.torchcell.sequence.construct.test_construct_registry]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/sequence/construct/test_construct_registry.py
"""The construct index: identity by sha256, keys by label plus SO id plus span, refusals.

Every rule here is one the 17 mirrored Cachera maps broke in the audit's design, so each
test states the measurement it defends. The fixtures are hand-written maps rather than
the mirror, so the suite runs on a machine with no DATA_ROOT; the mirror-backed
assertions live in the deposit script, which re-measures on every run.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from torchcell.artifacts.ref import ArtifactRef
from torchcell.datamodels.schema import SOTerm
from torchcell.literature.manifest import (
    MANIFEST_FILENAME,
    ArtifactRecord,
    Manifest,
    RetrievalMethod,
    RetrievalRecord,
)
from torchcell.sequence.construct import (
    TOPOLOGY_SO,
    BlockedConstruct,
    BlockedReason,
    ConstructContig,
    ConstructFeature,
    ConstructIntegrityError,
    ConstructKind,
    ConstructManifest,
    ConstructSpan,
    FeatureKey,
    PlacementTargetError,
    SequenceStatus,
    SpanComposition,
    construct_set_dir,
    deposit_construct_set,
    extend_construct_set,
    load_construct_manifest,
    parse_contig_features,
    resolve_contig,
    verify_construct_set,
)
from torchcell.sequence.plasmid import GenBankLocationError, LocationRefusal
from torchcell.verification.report import Provenance
from torchcell.verification.sourced import (
    ProvenanceGap,
    ProvenanceGapReason,
    SourcedValue,
)

KEY = "testConstructKey2026"
SET = "test_construct_set"
GENE = SOTerm(so_id="SO:0000704", name="gene")
PRIMER = SOTerm(so_id="SO:0005850", name="primer_binding_site")

#: A 20 bp circular map: a promoter, a reverse CDS, and a gene/CDS pair at ONE span,
#: which is the collision measured in pHO22 (``Ampicillin`` as both a gene and a CDS at
#: 1543..2403) and the reason the SO id is part of the key.
_MAP = """\
LOCUS       pFIX                      20 bp    DNA     circular SYN 01-JAN-2026
DEFINITION  fixture.
ACCESSION   pFIX
VERSION     pFIX
FEATURES             Location/Qualifiers
     source          1..20
                     /organism="synthetic DNA construct"
     promoter        1..4
                     /label="pTEF"
     CDS             complement(5..10)
                     /gene="KanR"
     gene            11..16
                     /label="AmpR"
     CDS             11..16
                     /label="AmpR"
     primer          17..20
                     /label="HOP1"
ORIGIN
        1 atgcgtacgt acgtacgtac
//
"""


def _write_map(root: Path, name: str = "si/plasmids/pFIX.gb") -> tuple[Path, str, int]:
    """Place the fixture map in a library-shaped tier and return (path, sha256, bytes)."""
    path = root / "torchcell-library" / KEY / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_MAP)
    data = path.read_bytes()
    return path, hashlib.sha256(data).hexdigest(), len(data)


def _write_tier_manifest(root: Path, records: list[ArtifactRecord]) -> None:
    key_root = root / "torchcell-library" / KEY
    manifest = Manifest(citation_key=KEY, files=records, created_at="2026-10-15")
    (key_root / MANIFEST_FILENAME).write_text(manifest.model_dump_json(indent=2))


def _retrieval(sha256: str) -> RetrievalRecord:
    return RetrievalRecord(
        method=RetrievalMethod.manual_browser,
        source_url="https://example.invalid/article",
        retriever="manual",
        params={"retrieval_command": "download the supplementary archive by hand"},
        sha256=sha256,
        retrieved_at="2026-10-15",
    )


def _features(root: Path) -> list[ConstructFeature]:
    path, _, _ = _write_map(root)
    _, features = parse_contig_features(path, KEY)
    return features


def _contig(
    root: Path, *, with_retrieval: bool = True
) -> tuple[ConstructContig, ArtifactRecord]:
    path, sha256, size = _write_map(root)
    component, features = parse_contig_features(path, KEY)
    record = ArtifactRecord(
        path="si/plasmids/pFIX.gb",
        role="si_data",
        bytes=size,
        sha256=sha256,
        retrieval=_retrieval(sha256) if with_retrieval else None,
    )
    contig = ConstructContig(
        sha256=sha256,
        ref=ArtifactRef(
            tier="library", key=KEY, path=record.path, sha256=sha256, bytes=size
        ),
        record_name=component.identity,
        topology="circular",
        length=component.length,
        kind=ConstructKind.plasmid,
        features=features,
        source_name=SourcedValue(
            value="pFix1",
            provenance=Provenance(
                source_uri="paper.md", citation_key=KEY, sha256="a" * 64
            ),
            quote="the fixture plasmid pFix1",
        ),
    )
    return contig, record


def _manifest(
    contig: ConstructContig, record: ArtifactRecord, **kwargs: object
) -> ConstructManifest:
    base: dict[str, object] = {
        "construct_set": SET,
        "source": "a fixture",
        "sequence_status": SequenceStatus.design,
        "citation_key": KEY,
        "files": [record],
        "contigs": [contig],
        "provenance_complete": record.retrieval is not None,
        "created_at": "2026-10-15T00:00:00+00:00",
    }
    base.update(kwargs)
    return ConstructManifest(**base)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Identity and keys
# --------------------------------------------------------------------------- #
def test_the_feature_key_needs_the_so_id_because_label_and_span_collide(
    tmp_path: Path,
) -> None:
    """A gene and a CDS at one span share a label and a span, so the key is the triple.

    Measured on the mirror: label alone collides in 4 of 17 maps, label plus span still
    collides in 1 of 17 (pHO22's ``Ampicillin`` gene/CDS pair at 1543..2403), and label
    plus SO id plus span is unique in all 17. The fixture reproduces that pair.
    """
    features = _features(tmp_path)
    amp = [f for f in features if f.key.label == "AmpR"]
    assert len(amp) == 2
    assert {f.key.span.start for f in amp} == {11}
    assert {f.key.so_id for f in amp} == {"SO:0000704", "SO:0000316"}
    tokens = [f.key.token for f in features]
    assert len(set(tokens)) == len(tokens)


def test_spans_are_one_based_end_inclusive(tmp_path: Path) -> None:
    """GenBank ``1..4`` becomes ``1..4``, not the parser's 0-based half-open ``[0, 4)``.

    ``GenomicSpan`` is 1-based end-inclusive, and one store serving native and construct
    genes must not carry two conventions, so the conversion happens at this boundary.
    """
    promoter = next(f for f in _features(tmp_path) if f.key.label == "pTEF")
    assert (promoter.key.span.start, promoter.key.span.end) == (1, 4)
    assert promoter.key.span.length == 4
    assert promoter.key.span.orientation == "inline"


def test_a_reverse_feature_keeps_its_orientation(tmp_path: Path) -> None:
    """``complement(5..10)`` keeps the reverse orientation in its key."""
    kanr = next(f for f in _features(tmp_path) if f.key.label == "KanR")
    assert (kanr.key.span.start, kanr.key.span.end) == (5, 10)
    assert kanr.key.span.orientation == "reverse_complement"
    assert kanr.key.token.endswith("-")


def test_a_contig_handle_is_the_sha256_prefix_not_the_locus_name(
    tmp_path: Path,
) -> None:
    """Identity is the file's sha256; the LOCUS line is data.

    ``pBTX002.gb`` carries ``LOCUS pBTX001`` and ``pBTX001(ante113).gb`` carries
    ``LOCUS ANTE113_with_NatMX``, so a LOCUS-derived id names the wrong replicon. The
    handle is the first 12 hex of the hash and ``record_name`` keeps the LOCUS verbatim.
    """
    contig, _ = _contig(tmp_path)
    assert contig.handle == contig.sha256[:12]
    assert len(contig.handle) == 12
    assert contig.record_name == "pFIX"


def test_a_contig_whose_ref_pins_other_bytes_is_refused(tmp_path: Path) -> None:
    """The contig's sha256 and its ref's must be the same hash: identity IS the hash."""
    contig, _ = _contig(tmp_path)
    payload = contig.model_dump()
    payload["ref"] = payload["ref"] | {"sha256": "b" * 64}
    with pytest.raises(ValidationError, match="disagrees with its ref"):
        ConstructContig.model_validate(payload)


def test_a_contig_with_an_unsourced_name_is_refused(tmp_path: Path) -> None:
    """``source_name`` is a quoted value or a typed gap, never a silent None."""
    contig, _ = _contig(tmp_path)
    payload = contig.model_dump()
    payload["source_name"] = None
    with pytest.raises(ValidationError, match="source_name is unset"):
        ConstructContig.model_validate(payload)


def test_a_contig_may_gap_its_name_with_a_reason(tmp_path: Path) -> None:
    """A construct whose name no mirrored artifact states carries the typed absence.

    ``not_released_by_primary`` is the construct-store member added for exactly this:
    Kang 2026's sequences are "upon request" and Carruthers 2025's are verified but not
    deposited, so neither the value nor an artifact of ours holds the name.
    """
    contig, _ = _contig(tmp_path)
    payload = contig.model_dump()
    payload["source_name"] = None
    payload["provenance_gaps"] = [
        ProvenanceGap(
            field="source_name",
            reason=ProvenanceGapReason.not_released_by_primary,
            looked_in=Provenance(source_uri="paper.md", citation_key=KEY),
        ).model_dump()
    ]
    gapped = ConstructContig.model_validate(payload)
    assert gapped.source_name is None
    assert gapped.provenance_gaps[0].reason == "not_released_by_primary"


def test_a_contig_cannot_both_state_and_gap_its_name(tmp_path: Path) -> None:
    """A gapped field must be None: never both a value and a declared absence."""
    contig, _ = _contig(tmp_path)
    payload = contig.model_dump()
    payload["provenance_gaps"] = [
        ProvenanceGap(
            field="source_name", reason=ProvenanceGapReason.not_released_by_primary
        ).model_dump()
    ]
    with pytest.raises(ValidationError, match="is not None"):
        ConstructContig.model_validate(payload)


def test_repeated_feature_keys_within_a_contig_are_refused(tmp_path: Path) -> None:
    """A key that is not unique within its contig is no key at all."""
    contig, _ = _contig(tmp_path)
    duplicate = contig.features[0]
    payload = contig.model_dump()
    payload["features"] = [*payload["features"], duplicate.model_dump()]
    with pytest.raises(ValidationError, match="feature keys repeat"):
        ConstructContig.model_validate(payload)


# --------------------------------------------------------------------------- #
# Placement targets
# --------------------------------------------------------------------------- #
def test_a_primer_is_indexed_but_refused_as_a_placement_target(tmp_path: Path) -> None:
    """297 of 570 features in the Cachera set are primers, and none may be a target.

    The data is the data, so a primer IS indexed: it is real annotation of the canonical
    bytes. What is refused is pointing a placement at one, because a primer binding site
    is an oligo's footprint on the map rather than a part of the construct, and the
    refusal keeps the modal feature of the store from being a legal target.
    """
    contig, _ = _contig(tmp_path)
    primer = next(f for f in contig.features if f.key.label == "HOP1")
    assert primer.key.so_id == PRIMER.so_id
    assert primer.key.is_placement_target is False
    assert contig.feature(primer.key.token) is primer
    with pytest.raises(PlacementTargetError, match="oligo footprint"):
        contig.placement_target(primer.key.token)


def test_a_gene_is_a_placement_target(tmp_path: Path) -> None:
    """Every non-primer feature stays targetable; the refusal is one SO id, not a rule."""
    contig, _ = _contig(tmp_path)
    promoter = next(f for f in contig.features if f.key.label == "pTEF")
    assert promoter.key.is_placement_target is True
    assert contig.placement_target(promoter.key.token) is promoter


def test_an_unknown_feature_token_raises(tmp_path: Path) -> None:
    """A token the contig does not hold raises instead of returning None."""
    contig, _ = _contig(tmp_path)
    with pytest.raises(KeyError):
        contig.feature("nothing@SO:0000704:1-2+")


def test_a_feature_key_must_be_a_well_formed_so_id() -> None:
    """The key's SO id goes through ``SOTerm``'s pattern: one SO map for the repo."""
    with pytest.raises(ValidationError):
        FeatureKey(label="x", so_id="SO:704", span=ConstructSpan.one(1, 2, "inline"))


def test_a_span_must_be_one_based_and_ordered() -> None:
    """``ConstructSpan`` refuses a 0 start and an inverted interval."""
    with pytest.raises(ValidationError):
        ConstructSpan.one(0, 4, "inline")
    with pytest.raises(ValidationError):
        ConstructSpan.one(9, 4, "inline")


def test_a_feature_key_whose_so_id_is_not_among_its_roles_is_refused() -> None:
    """The key's SO id names one of the feature's own roles, not an unrelated term."""
    with pytest.raises(ValidationError, match="roles are"):
        ConstructFeature(
            key=FeatureKey(
                label="x", so_id="SO:0000704", span=ConstructSpan.one(1, 4, "inline")
            ),
            roles=[PRIMER],
            genbank_type="primer",
        )


# --------------------------------------------------------------------------- #
# Deposit, resolve, extend
# --------------------------------------------------------------------------- #
def test_deposit_then_resolve_rehashes_the_bytes(tmp_path: Path) -> None:
    """Deposit re-hashes everything it records; resolve re-hashes on every call."""
    contig, record = _contig(tmp_path)
    _write_tier_manifest(tmp_path, [record])
    path = deposit_construct_set(_manifest(contig, record), str(tmp_path))
    assert Path(path).is_file()
    loaded = load_construct_manifest(SET, str(tmp_path))
    assert [c.handle for c in loaded.contigs] == [contig.handle]
    assert loaded.provenance_complete is True
    resolved = resolve_contig(SET, contig.handle, data_root=str(tmp_path))
    assert resolved.read_text() == _MAP
    assert verify_construct_set(SET, str(tmp_path)) == {contig.handle: contig.sha256}


def test_deposit_refuses_to_overwrite_an_index(tmp_path: Path) -> None:
    """A deposited index is never rewritten: a stored record may point at its hashes."""
    contig, record = _contig(tmp_path)
    _write_tier_manifest(tmp_path, [record])
    deposit_construct_set(_manifest(contig, record), str(tmp_path))
    with pytest.raises(FileExistsError, match="never rewritten"):
        deposit_construct_set(_manifest(contig, record), str(tmp_path))


def test_deposit_refuses_a_length_the_bytes_do_not_have(tmp_path: Path) -> None:
    """A length is re-measured from the bytes; a header-copied claim cannot slip in."""
    contig, record = _contig(tmp_path)
    _write_tier_manifest(tmp_path, [record])
    lying = ConstructContig.model_validate(contig.model_dump() | {"length": 999})
    with pytest.raises(ConstructIntegrityError, match="the index says 999"):
        deposit_construct_set(_manifest(lying, record), str(tmp_path))


def test_resolve_detects_bytes_that_changed_after_deposit(tmp_path: Path) -> None:
    """Changed bytes raise on resolve rather than being served as the pinned sequence."""
    contig, record = _contig(tmp_path)
    _write_tier_manifest(tmp_path, [record])
    deposit_construct_set(_manifest(contig, record), str(tmp_path))
    target = tmp_path / "torchcell-library" / KEY / "si/plasmids/pFIX.gb"
    target.write_text(_MAP.replace("pTEF", "pGAL"))
    with pytest.raises(Exception) as drift:
        resolve_contig(SET, contig.handle, data_root=str(tmp_path))
    assert "sha256" in str(drift.value)


def test_provenance_complete_is_refused_without_a_retrieval_record(
    tmp_path: Path,
) -> None:
    """Completeness is per file: one file with no retrieval record and it is False.

    The key manifest's own ``provenance_complete: True`` is a statement about the KEY,
    not evidence a construct's retrieval reproduces; every one of the 17 Cachera files
    has ``retrieval: null`` there. The index earns the claim per file or not at all.
    """
    contig, record = _contig(tmp_path, with_retrieval=False)
    with pytest.raises(ValidationError, match="carry no RetrievalRecord"):
        _manifest(contig, record, provenance_complete=True)
    incomplete = _manifest(contig, record, provenance_complete=False)
    assert incomplete.provenance_complete is False


def test_a_contig_whose_file_the_index_does_not_list_is_refused(tmp_path: Path) -> None:
    """A contig points at a file the index lists, so the index is self-contained."""
    contig, record = _contig(tmp_path)
    with pytest.raises(ValidationError, match="does not list"):
        _manifest(contig, record, files=[])


def test_blocked_maps_are_listed_by_name_with_the_refusal(tmp_path: Path) -> None:
    """A map the parser refuses stays in ``files`` and is named in ``blocked``.

    This is what "report, do not hide" means for a map: 11 of the 17 Cachera maps carry
    an origin-wrapping location, the paper's pBTX2 among them, and each one keeps its
    retrieval record and its sha256 while saying exactly what stopped it.
    """
    contig, record = _contig(tmp_path)
    other = ArtifactRecord(
        path="si/plasmids/pWRAP.gb",
        role="si_data",
        bytes=10,
        sha256="c" * 64,
        retrieval=_retrieval("c" * 64),
    )
    manifest = _manifest(
        contig,
        record,
        files=[record, other],
        blocked=[
            BlockedConstruct(
                path=other.path,
                reason=BlockedReason.origin_wrapping_location,
                detail="pWRAP.gb: Biopython rewrote an origin-wrapping location",
            )
        ],
    )
    assert [entry.path for entry in manifest.blocked] == ["si/plasmids/pWRAP.gb"]
    assert manifest.record(other.path).sha256 == "c" * 64
    with pytest.raises(ValidationError, match="is not listed"):
        _manifest(
            contig,
            record,
            blocked=[
                BlockedConstruct(
                    path="si/plasmids/absent.gb",
                    reason=BlockedReason.compound_location,
                    detail="x",
                )
            ],
        )


def test_extend_is_additive_and_refuses_a_listed_path(tmp_path: Path) -> None:
    """Adding changes no existing claim; re-recording a listed file is refused."""
    contig, record = _contig(tmp_path)
    _write_tier_manifest(tmp_path, [record])
    deposit_construct_set(_manifest(contig, record), str(tmp_path))
    added = ArtifactRecord(
        path="si/plasmids/pLATER.gb",
        role="si_data",
        bytes=12,
        sha256="d" * 64,
        retrieval=_retrieval("d" * 64),
    )
    extend_construct_set(
        SET,
        [added],
        new_blocked=[
            BlockedConstruct(
                path=added.path,
                reason=BlockedReason.compound_location,
                detail="pLATER.gb: feature gene 'x' has the compound location",
            )
        ],
        data_root=str(tmp_path),
    )
    grown = load_construct_manifest(SET, str(tmp_path))
    assert [r.path for r in grown.files] == [record.path, added.path]
    assert [c.handle for c in grown.contigs] == [contig.handle]
    assert len(grown.blocked) == 1
    with pytest.raises(FileExistsError, match="already in the index"):
        extend_construct_set(SET, [added], data_root=str(tmp_path))
    with pytest.raises(ValueError, match="nothing to add"):
        extend_construct_set(SET, [], data_root=str(tmp_path))


def test_extend_recomputes_completeness(tmp_path: Path) -> None:
    """A file with no retrieval record arriving makes a complete index incomplete."""
    contig, record = _contig(tmp_path)
    _write_tier_manifest(tmp_path, [record])
    deposit_construct_set(_manifest(contig, record), str(tmp_path))
    assert load_construct_manifest(SET, str(tmp_path)).provenance_complete is True
    extend_construct_set(
        SET,
        [
            ArtifactRecord(
                path="si/plasmids/pNOPROV.gb", role="si_data", bytes=5, sha256="e" * 64
            )
        ],
        data_root=str(tmp_path),
    )
    assert load_construct_manifest(SET, str(tmp_path)).provenance_complete is False


def test_an_index_naming_another_set_is_refused(tmp_path: Path) -> None:
    """The directory and the manifest must name the same set."""
    contig, record = _contig(tmp_path)
    _write_tier_manifest(tmp_path, [record])
    deposit_construct_set(_manifest(contig, record), str(tmp_path))
    path = Path(construct_set_dir(SET, str(tmp_path))) / MANIFEST_FILENAME
    payload = json.loads(path.read_text())
    payload["construct_set"] = "another_set"
    path.write_text(json.dumps(payload))
    with pytest.raises(ConstructIntegrityError, match="names construct set"):
        load_construct_manifest(SET, str(tmp_path))


def test_an_absent_index_names_the_provisioning_script(tmp_path: Path) -> None:
    """A machine without the set fails with the command that creates it."""
    with pytest.raises(FileNotFoundError, match="provision_construct_sets"):
        load_construct_manifest("no_such_set", str(tmp_path))


def test_an_origin_wrapping_map_now_indexes_with_its_parts(tmp_path: Path) -> None:
    """``parse_contig_features`` projects a wrap into a two-part 1-based span.

    ``19..2`` on this 20 bp circular fixture is the form 11 of the 17 Cachera maps carry
    (``11559..128``, ``complement(4892..858)``). It used to block the map; it is now the
    ordered parts ``19..20`` and ``1..2``, which is 4 bases, not the 20 the envelope
    would have given, and the key's token carries the ``wrap:`` sigil so it can never
    collide with a join over the same two intervals.
    """
    path = tmp_path / "pWRAP.gb"
    path.write_text(
        _MAP.replace("     primer          17..20\n", "     primer          19..2\n")
    )
    _, features = parse_contig_features(path, KEY)
    wrap = next(f for f in features if f.key.label == "HOP1")
    assert wrap.key.span.composition is SpanComposition.origin_spanning
    assert [(p.start, p.end) for p in wrap.key.span.parts] == [(19, 20), (1, 2)]
    assert wrap.key.span.length == 4
    assert wrap.key.token == "HOP1@SO:0005850:wrap:19-20,1-2+"
    assert wrap.key.span.insdc_location(compact_wrap=True) == "19..2"
    assert wrap.key.span.insdc_location() == "join(19..20,1..2)"
    with pytest.raises(ValueError, match="no single start"):
        _ = wrap.key.span.start


def test_a_wrap_is_checked_against_the_contig_length_and_topology(
    tmp_path: Path,
) -> None:
    """An ``origin_spanning`` claim is proved on the contig, which knows the bytes.

    ``ConstructSpan`` can check the shape of its parts but not their relation to the
    replicon: the first part of a wrap must end at the LAST base, and a linear contig
    has no origin to cross. Both are refused here rather than at the span, because this
    is where the length and the topology live.
    """
    path = tmp_path / "pWRAP.gb"
    path.write_text(
        _MAP.replace("     primer          17..20\n", "     primer          19..2\n")
    )
    component, features = parse_contig_features(path, KEY)
    base = {
        "sha256": "f" * 64,
        "ref": ArtifactRef(
            tier="library",
            key=KEY,
            path="si/plasmids/pWRAP.gb",
            sha256="f" * 64,
            bytes=10,
        ).model_dump(),
        "record_name": component.identity,
        "length": component.length,
        "kind": ConstructKind.plasmid,
        "features": [f.model_dump() for f in features],
        "source_name": SourcedValue(
            value="pWrap1",
            provenance=Provenance(
                source_uri="paper.md", citation_key=KEY, sha256="a" * 64
            ),
            quote="the fixture plasmid pWrap1",
        ).model_dump(),
    }
    assert ConstructContig.model_validate(base | {"topology": "circular"})
    with pytest.raises(ValidationError, match="no origin to cross"):
        ConstructContig.model_validate(base | {"topology": "linear"})
    with pytest.raises(ValidationError, match="past the contig's"):
        ConstructContig.model_validate(
            base | {"topology": "circular", "length": component.length - 1}
        )


def test_a_blocked_map_records_the_reason_the_parser_gave(tmp_path: Path) -> None:
    """The index's reason comes from ``GenBankLocationError.refusal``, not the call site.

    All 11 Cachera refusals were recorded as ``origin_wrapping_location`` because that
    member was hardcoded where the refusal was caught. One vocabulary, carried on the
    exception, is what keeps the index's reason the parser's own: an INSDC ``order``
    location is an ``ordered_location`` and says so.
    """
    path = tmp_path / "pORDER.gb"
    path.write_text(
        _MAP.replace(
            "     primer          17..20\n",
            "     primer          order(17..18,19..20)\n",
        )
    )
    with pytest.raises(GenBankLocationError) as caught:
        parse_contig_features(path, KEY)
    assert caught.value.refusal is BlockedReason.ordered_location
    assert BlockedReason is LocationRefusal


def test_a_contig_grounds_its_topology_in_the_sequence_ontology(tmp_path: Path) -> None:
    """``topology`` reads out as an SO term; the map is derived, never a second field.

    SO's ``topology_attribute`` (SO:0000986) has exactly the two children a replicon's
    topology can be, which is why the Literal has two members and the grounding is a
    lookup rather than a stored term.
    """
    contig, _ = _contig(tmp_path)
    assert contig.topology == "circular"
    assert contig.topology_term == SOTerm(so_id="SO:0000988", name="circular")
    assert set(TOPOLOGY_SO) == {"circular", "linear"}


def test_the_module_imports_first_in_a_fresh_interpreter() -> None:
    """The construct package can be the first torchcell import of a process.

    It reaches ``schema`` (for ``SOTerm``) and ``artifacts.resolve`` (the ONE resolver),
    both of which sit near the 2026-10-07 import cycle, so the order is pinned in a
    subprocess where nothing pytest imported can mask it.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import torchcell.sequence.construct as c\nprint(c.CONSTRUCTS_DIR)",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip() == "torchcell-constructs"
