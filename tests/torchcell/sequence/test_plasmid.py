# tests/torchcell/sequence/test_plasmid.py
"""Unit tests for the WS10 plasmid sequence store (feature extraction + strand + wrap)."""

from __future__ import annotations

import pytest

from torchcell.sequence.plasmid import (
    PlasmidFeature,
    PlasmidProvenance,
    PlasmidSequence,
)

PROV = PlasmidProvenance(source_file="t.gb", sha256="0" * 64, citation_key="test")


def _plasmid(topology: str) -> PlasmidSequence:
    # seq:  A0 T1 G2 C3 G4 T5 A6 C7 G8 T9
    return PlasmidSequence(
        plasmid_id="p1",
        name="p1",
        topology=topology,
        length=10,
        sequence="ATGCGTACGT",
        features=[
            PlasmidFeature(feature_type="gene", label="fwd", start=0, end=3, strand=1),
            PlasmidFeature(feature_type="gene", label="rev", start=3, end=6, strand=-1),
            PlasmidFeature(feature_type="gene", label="edge", start=0, end=2, strand=1),
            PlasmidFeature(feature_type="gene", label="dup", start=0, end=1, strand=1),
            PlasmidFeature(feature_type="gene", label="dup", start=1, end=2, strand=1),
        ],
        provenance=PROV,
    )


def test_forward_feature_sequence():
    assert _plasmid("linear").feature_sequence("fwd") == "ATG"


def test_reverse_feature_is_reverse_complemented():
    # seq[3:6] == "CGT"; minus strand -> reverse complement -> "ACG"
    assert _plasmid("linear").feature_sequence("rev") == "ACG"


def test_flank_extends_and_clamps_on_linear():
    # fwd is [0,3); flank 2 -> [-2,5) clamps to [0,5) on a linear plasmid
    assert _plasmid("linear").feature_sequence("fwd", flank=2) == "ATGCG"


def test_circular_wrap():
    # edge is [0,2); flank 3 -> indices [-3..5) mod 10 = [7,8,9,0,1,2,3,4]
    assert _plasmid("circular").feature_sequence("edge", flank=3) == "CGTATGCG"


def test_missing_feature_raises():
    with pytest.raises(KeyError):
        _plasmid("linear").get_feature("nope")


def test_duplicate_feature_raises():
    with pytest.raises(KeyError):
        _plasmid("linear").get_feature("dup")


def test_round_trip():
    p = _plasmid("circular")
    assert p == PlasmidSequence(**p.model_dump())
