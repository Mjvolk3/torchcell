# tests/torchcell/sequence/test_plasmid.py
"""Unit tests for the WS10 SBOL-aligned plasmid Component (extraction + composition)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from pydantic import ValidationError

from torchcell.datamodels.schema import SOTerm
from torchcell.sequence.plasmid import (
    Component,
    Feature,
    GenBankLocationError,
    Location,
    SequenceProvenance,
    SORole,
    _sha256,
    parse_genbank_component,
)

PROV = SequenceProvenance(source_file="t.gb", sha256="0" * 64, citation_key="test")
GENE = SORole(so_id="SO:0000704", name="gene")
PROMOTER = SORole(so_id="SO:0000167", name="promoter")


def _loc(start: int, end: int, rev: bool = False) -> Location:
    return Location(
        start=start, end=end, orientation="reverse_complement" if rev else "inline"
    )


def _component(topology: str) -> Component:
    # seq:  A0 T1 G2 C3 G4 T5 A6 C7 G8 T9
    return Component(
        identity="p1",
        roles=[SORole(so_id="SO:0000155", name="plasmid_vector")],
        topology=topology,
        length=10,
        sequence="ATGCGTACGT",
        features=[
            Feature(name="fwd", roles=[GENE], location=_loc(0, 3)),
            Feature(name="rev", roles=[GENE], location=_loc(3, 6, rev=True)),
            Feature(name="edge", roles=[GENE], location=_loc(0, 2)),
            Feature(name="promA", roles=[PROMOTER], location=_loc(6, 8)),
            Feature(name="dup", roles=[GENE], location=_loc(0, 1)),
            Feature(name="dup", roles=[GENE], location=_loc(1, 2)),
        ],
        provenance=PROV,
    )


def test_forward_feature_sequence():
    assert _component("linear").feature_sequence("fwd") == "ATG"


def test_reverse_feature_is_reverse_complemented():
    # seq[3:6] == "CGT"; reverse strand -> reverse complement -> "ACG"
    assert _component("linear").feature_sequence("rev") == "ACG"


def test_flank_extends_and_clamps_on_linear():
    assert _component("linear").feature_sequence("fwd", flank=2) == "ATGCG"


def test_circular_wrap():
    # edge is [0,2); flank 3 -> indices [-3..5) mod 10 = [7,8,9,0,1,2,3,4]
    assert _component("circular").feature_sequence("edge", flank=3) == "CGTATGCG"


def test_missing_feature_raises():
    with pytest.raises(KeyError):
        _component("linear").get_feature("nope")


def test_duplicate_feature_raises():
    with pytest.raises(KeyError):
        _component("linear").get_feature("dup")


def test_features_by_role():
    c = _component("linear")
    assert [f.name for f in c.features_by_role("SO:0000167")] == ["promA"]
    assert {f.name for f in c.features_by_role("SO:0000704")} == {
        "fwd",
        "rev",
        "edge",
        "dup",
    }


def test_subcomponent_carves_and_rebases():
    # carve [2,8) -> seq "GCGTAC" (len 6); features fully inside: rev [3,6)->[1,4),
    # promA [6,8)->[4,6). fwd/edge (start 0) excluded.
    sub = _component("linear").subcomponent(2, 8, "insert")
    assert sub.length == 6 and sub.sequence == "GCGTAC" and sub.topology == "linear"
    names = {f.name: (f.location.start, f.location.end) for f in sub.features}
    assert names == {"rev": (1, 4), "promA": (4, 6)}
    # extraction on the carved Component still works (rev-strand feature)
    assert sub.feature_sequence("rev") == "ACG"


def test_round_trip():
    c = _component("circular")
    assert c == Component(**c.model_dump())


# --------------------------------------------------------------------------- #
# 2026.10.06 (Phase 21): GenBank ingestion of a hand-written 20 bp circular map.
# Features: promoter 1..4 (label pTEF), CDS complement(5..10) (gene KanR, no label),
# rep_origin 11..14 (no name), misc_binding 15..16 (an unmapped type), and gene
# join(18..20,1..2) (label wrap) that spans the origin. GenBank is 1-based closed, so
# 1..4 is half-open [0, 4).
# --------------------------------------------------------------------------- #
_GENBANK = """\
LOCUS       pTEST                     20 bp    DNA     circular SYN 01-JAN-2026
DEFINITION  test plasmid.
ACCESSION   pTEST
VERSION     pTEST
KEYWORDS    .
SOURCE      synthetic DNA construct
  ORGANISM  synthetic DNA construct
            other sequences.
FEATURES             Location/Qualifiers
     source          1..20
                     /organism="synthetic DNA construct"
     promoter        1..4
                     /label="pTEF"
     CDS             complement(5..10)
                     /gene="KanR"
     rep_origin      11..14
     misc_binding    15..16
                     /label="odd"
     gene            join(18..20,1..2)
                     /label="wrap"
ORIGIN
        1 atgcgtacgt acgtacgtac
//
"""


# The same map WITHOUT the origin-spanning feature. Since 2026-10-15 the parser refuses
# a wrapping or compound location instead of flattening it, so the happy path needs a
# map whose every feature is one range; the wrap stays in ``_GENBANK`` because it is
# what the refusal tests are about.
_GENBANK_SINGLE_RANGE = _GENBANK.replace(
    '     gene            join(18..20,1..2)\n                     /label="wrap"\n', ""
)


def _gb(tmp_path: Path) -> Path:
    path = tmp_path / "pTEST.gb"
    path.write_text(_GENBANK)
    return path


def _gb_single_range(tmp_path: Path) -> Path:
    path = tmp_path / "pSINGLE.gb"
    path.write_text(_GENBANK_SINGLE_RANGE)
    return path


def test_parse_genbank_component_features_roles_and_provenance(tmp_path: Path) -> None:
    path = _gb_single_range(tmp_path)
    comp = parse_genbank_component(str(path), "testKey2026")
    assert (comp.identity, comp.topology, comp.length) == ("pTEST", "circular", 20)
    assert comp.sequence == "ATGCGTACGTACGTACGTAC"
    assert comp.roles == [SORole(so_id="SO:0000155", name="plasmid_vector")]
    assert comp.provenance == SequenceProvenance(
        source_file="pSINGLE.gb",
        sha256=hashlib.sha256(_GENBANK_SINGLE_RANGE.encode()).hexdigest(),
        citation_key="testKey2026",
    )
    got = [
        (
            f.name,
            f.roles[0].so_id,
            f.location.start,
            f.location.end,
            f.location.orientation,
        )
        for f in comp.features
    ]
    assert got[:4] == [
        ("pTEF", "SO:0000167", 0, 4, "inline"),
        ("KanR", "SO:0000316", 4, 10, "reverse_complement"),
        ("", "SO:0000296", 10, 14, "inline"),
        ("odd", "SO:0000110", 14, 16, "inline"),
    ]
    # [4, 10) is GTACGT; reversed TGCATG; complemented ACGTAC
    assert comp.feature_sequence("KanR") == "ACGTAC"


def test_parse_genbank_component_refuses_an_origin_spanning_compound_location(
    tmp_path: Path,
) -> None:
    """The origin-spanning ``join(18..20,1..2)`` is REFUSED, not flattened (2026-10-15).

    Until 2026-10-15 a compound location was reduced to ``[min start, max end)``, so
    this 5 bp feature (``TAC`` + ``AT``) was stored as ``[0, 20)`` -- the WHOLE plasmid
    -- and ``feature_sequence("wrap")`` returned all 20 bp. That was a warn-and-fix
    fallback on a case plasmid maps carry routinely. The refusal names the file and the
    feature, which is how a map carrying one is reported by name instead of being
    indexed with a false coordinate. Biopython accepts an EXPLICIT join across the
    origin without complaint, so this case is caught by the parts check, not by the
    warning check the next test covers.
    """
    with pytest.raises(GenBankLocationError) as refusal:
        parse_genbank_component(str(_gb(tmp_path)), "k")
    message = str(refusal.value)
    assert "pTEST.gb" in message
    assert "'wrap'" in message
    assert "compound location" in message


def test_parse_genbank_component_refuses_a_location_biopython_rewrites(
    tmp_path: Path,
) -> None:
    """A single-range wrapping location Biopython "fixes" is refused, not accepted.

    ``18..2`` on a 20 bp circular replicon is the form the mirrored maps carry (11 of
    17 Cachera maps have one, e.g. ``4892..858``): Biopython warns ``Attempting to fix
    invalid location ... incorrect origin wrapping`` and hands back a two-part location
    whose envelope is the whole plasmid. Taking that silently is the fallback this
    refusal exists to prevent, so the parse runs under ``catch_warnings`` and raises
    with the offending location in the message.
    """
    text = _GENBANK.replace(
        "     gene            join(18..20,1..2)\n", "     gene            18..2\n"
    )
    path = tmp_path / "pWRAP.gb"
    path.write_text(text)
    with pytest.raises(GenBankLocationError) as refusal:
        parse_genbank_component(str(path), "k")
    message = str(refusal.value)
    assert "pWRAP.gb" in message
    assert "18..2" in message
    assert "origin-wrapping" in message


def test_so_role_is_the_one_so_term_class() -> None:
    """``plasmid.SORole`` IS ``schema.SOTerm``: one Sequence Ontology map for the repo.

    The module used to define a second class with the same two fields, which is two
    classes asserting one fact. The alias also means a malformed id now raises, which
    the local duplicate never checked.
    """
    assert SORole is SOTerm
    with pytest.raises(ValidationError):
        SORole(so_id="SO:704", name="gene")


def test_sha256_streams_more_than_one_block(tmp_path: Path) -> None:
    data = bytes(range(256)) * 9000  # 2,304,000 bytes: three 1 MiB reads
    path = tmp_path / "big.bin"
    path.write_bytes(data)
    assert _sha256(str(path)) == hashlib.sha256(data).hexdigest()
