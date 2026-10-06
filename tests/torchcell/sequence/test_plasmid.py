# tests/torchcell/sequence/test_plasmid.py
"""Unit tests for the WS10 SBOL-aligned plasmid Component (extraction + composition)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from torchcell.sequence.plasmid import (
    Component,
    Feature,
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


def _gb(tmp_path: Path) -> Path:
    path = tmp_path / "pTEST.gb"
    path.write_text(_GENBANK)
    return path


def test_parse_genbank_component_features_roles_and_provenance(tmp_path: Path) -> None:
    path = _gb(tmp_path)
    comp = parse_genbank_component(str(path), "testKey2026")
    assert (comp.identity, comp.topology, comp.length) == ("pTEST", "circular", 20)
    assert comp.sequence == "ATGCGTACGTACGTACGTAC"
    assert comp.roles == [SORole(so_id="SO:0000155", name="plasmid_vector")]
    assert comp.provenance == SequenceProvenance(
        source_file="pTEST.gb",
        sha256=hashlib.sha256(_GENBANK.encode()).hexdigest(),
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


def test_parse_genbank_component_flattens_an_origin_spanning_feature(
    tmp_path: Path,
) -> None:
    """Finding: a compound location is reduced to ``[min start, max end)``, so the
    origin-spanning ``join(18..20,1..2)`` (5 bp: ``TAC`` + ``AT``) is stored as
    ``[0, 20)``, the WHOLE plasmid, and its extracted sequence is all 20 bp. Plasmid
    maps routinely carry features across the origin. Pinned until compound/wrapping
    locations are kept (plasmid.py:200-208). Reach: nothing calls
    ``parse_genbank_component`` yet, so latent.
    """
    comp = parse_genbank_component(str(_gb(tmp_path)), "k")
    wrap = comp.get_feature("wrap")
    assert (wrap.location.start, wrap.location.end) == (0, 20)
    assert comp.feature_sequence("wrap") == "ATGCGTACGTACGTACGTAC"


def test_sha256_streams_more_than_one_block(tmp_path: Path) -> None:
    data = bytes(range(256)) * 9000  # 2,304,000 bytes: three 1 MiB reads
    path = tmp_path / "big.bin"
    path.write_bytes(data)
    assert _sha256(str(path)) == hashlib.sha256(data).hexdigest()
