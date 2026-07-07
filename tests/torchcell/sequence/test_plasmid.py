# tests/torchcell/sequence/test_plasmid.py
"""Unit tests for the WS10 SBOL-aligned plasmid Component (extraction + composition)."""

from __future__ import annotations

import pytest

from torchcell.sequence.plasmid import (
    Component,
    Feature,
    Location,
    SequenceProvenance,
    SORole,
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
