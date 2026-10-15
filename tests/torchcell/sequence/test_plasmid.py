# tests/torchcell/sequence/test_plasmid.py
"""Unit tests for the WS10 SBOL-aligned plasmid Component (extraction + composition)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from pydantic import ValidationError

from torchcell.datamodels.schema import SOTerm, SpanComposition
from torchcell.sequence.plasmid import (
    RETIRED_LOCATION_REFUSALS,
    Component,
    Feature,
    GenBankLocationError,
    Location,
    LocationRefusal,
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
            Feature(name="fwd", roles=[GENE], locations=[_loc(0, 3)]),
            Feature(name="rev", roles=[GENE], locations=[_loc(3, 6, rev=True)]),
            Feature(name="edge", roles=[GENE], locations=[_loc(0, 2)]),
            Feature(name="promA", roles=[PROMOTER], locations=[_loc(6, 8)]),
            Feature(name="dup", roles=[GENE], locations=[_loc(0, 1)]),
            Feature(name="dup", roles=[GENE], locations=[_loc(1, 2)]),
            Feature(
                name="wrap",
                roles=[GENE],
                locations=[_loc(8, 10), _loc(0, 2)],
                composition=SpanComposition.origin_spanning,
            ),
            Feature(
                name="spliced",
                roles=[GENE],
                locations=[_loc(0, 2), _loc(4, 6)],
                composition=SpanComposition.joined,
            ),
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
        "wrap",
        "spliced",
    }


def test_subcomponent_carves_and_rebases():
    # carve [2,8) -> seq "GCGTAC" (len 6); features fully inside: rev [3,6)->[1,4),
    # promA [6,8)->[4,6). fwd/edge (start 0) excluded, and so are the two multi-range
    # features, because a feature is kept only when EVERY range is inside the window.
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


# The same map WITHOUT the origin-spanning feature. The happy-path test asserts the
# whole feature list positionally, so it reads a map whose every feature is one range;
# the wrap stays in ``_GENBANK`` because it is what the span-of-parts tests are about.
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


def test_an_explicit_origin_spanning_join_is_held_as_the_two_parts_it_wraps_into(
    tmp_path: Path,
) -> None:
    """``join(18..20,1..2)`` on a 20 bp circular map is ONE interval, held as two parts.

    Until 2026-10-15 a compound location was reduced to ``[min start, max end)``, so
    this 5 bp feature (``TAC`` + ``AT``) was stored as ``[0, 20)`` -- the WHOLE plasmid
    -- and from 2026-10-15 it was refused instead. Now it is held: the two ranges in the
    order the file writes them, ``origin_spanning`` because the first ends at the last
    base of a circular replicon and the second begins at the origin, and
    ``feature_sequence`` returns the 5 bp the feature covers rather than all 20.
    """
    comp = parse_genbank_component(str(_gb(tmp_path)), "k")
    wrap = comp.get_feature("wrap")
    assert wrap.composition is SpanComposition.origin_spanning
    assert [(r.start, r.end) for r in wrap.locations] == [(17, 20), (0, 2)]
    assert wrap.orientation == "inline"
    assert wrap.length == 5
    assert comp.feature_sequence("wrap") == "TACAT"
    with pytest.raises(ValueError, match="no single location"):
        _ = wrap.location


def test_the_compact_origin_wrapping_form_biopython_rewrites_is_held_too(
    tmp_path: Path,
) -> None:
    """``18..2`` is the form the mirrored maps carry, and it means the same claim.

    11 of the 17 Cachera maps write a wrap this way (``11559..128``,
    ``complement(4892..858)``); Biopython warns ``Attempting to fix invalid location
    ... incorrect origin wrapping`` and hands back the two parts. Both spellings reach
    the same stored parts, which is why no field records which one the source used.
    """
    text = _GENBANK.replace(
        "     gene            join(18..20,1..2)\n", "     gene            18..2\n"
    )
    path = tmp_path / "pWRAP.gb"
    path.write_text(text)
    comp = parse_genbank_component(str(path), "k")
    wrap = comp.get_feature("wrap")
    assert wrap.composition is SpanComposition.origin_spanning
    assert [(r.start, r.end) for r in wrap.locations] == [(17, 20), (0, 2)]
    assert comp.feature_sequence("wrap") == "TACAT"


def test_a_reverse_strand_wrap_keeps_the_order_the_file_wrote(tmp_path: Path) -> None:
    """``complement(18..2)`` stores ``18..20`` then ``1..2``, not Biopython's 5'->3' order.

    Biopython hands a reverse-strand compound location's parts back in feature order,
    which for this location is ``[(0, 2), (17, 20)]`` -- the reverse of what the file
    writes. Storing that order would regenerate ``complement(1..20)``-shaped nonsense,
    so the parser re-reverses, and the stored parts regenerate the source string. The
    extracted sequence is the reverse complement of the forward concatenation: ``TAC`` +
    ``AT`` is ``TACAT``, whose reverse complement is ``ATGTA``.
    """
    text = _GENBANK.replace(
        "     gene            join(18..20,1..2)\n",
        "     gene            complement(18..2)\n",
    )
    path = tmp_path / "pREV.gb"
    path.write_text(text)
    wrap = parse_genbank_component(str(path), "k").get_feature("wrap")
    assert wrap.orientation == "reverse_complement"
    assert [(r.start, r.end) for r in wrap.locations] == [(17, 20), (0, 2)]
    assert parse_genbank_component(str(path), "k").feature_sequence("wrap") == "ATGTA"


def test_a_plain_two_part_join_is_joined_and_not_origin_spanning(
    tmp_path: Path,
) -> None:
    """``join(3..6,10..14)`` is two intervals with sequence between them, not a wrap.

    The two claims are kept apart by the contig's own topology and length: a wrap's
    first part ends at the last base and its second begins at base 1, which this join
    does not do. ``feature_sequence`` concatenates only the parts, so the 5 bp of
    sequence between them is absent from the result.
    """
    text = _GENBANK.replace(
        "     gene            join(18..20,1..2)\n",
        "     gene            join(3..6,10..14)\n",
    )
    path = tmp_path / "pJOIN.gb"
    path.write_text(text)
    comp = parse_genbank_component(str(path), "k")
    spliced = comp.get_feature("wrap")
    assert spliced.composition is SpanComposition.joined
    assert [(r.start, r.end) for r in spliced.locations] == [(2, 6), (9, 14)]
    assert spliced.length == 9
    # [2, 6) is GCGT and [9, 14) is TACGT; the 3 bp between them are omitted.
    assert comp.feature_sequence("wrap") == "GCGTTACGT"


@pytest.mark.parametrize(
    ("location", "refusal"),
    [
        ("order(3..6,10..14)", LocationRefusal.ordered_location),
        ("<3..6", LocationRefusal.fuzzy_position),
        ("3..>6", LocationRefusal.fuzzy_position),
        ("3^4", LocationRefusal.between_bases_site),
        ("join(complement(3..6),10..14)", LocationRefusal.mixed_strand_location),
        ("J00194.1:3..6", LocationRefusal.remote_reference_location),
        ("join(10..14,3..6)", LocationRefusal.unordered_join),
        ("3.6", LocationRefusal.unparsable_location),
    ],
)
def test_the_locations_still_refused_each_carry_their_typed_reason(
    tmp_path: Path, location: str, refusal: LocationRefusal
) -> None:
    """Each remaining case raises with its own member, never a shared one.

    Holding parts admits three of the INSDC location forms and no more. ``order`` says
    nothing is implied about joining the elements, a fuzzy bound is not a coordinate, a
    between-bases site is not an interval, a mixed-strand join is not one feature, a
    remote reference is not this contig's bytes, and a backwards join would have to be
    reordered to be stored, which is the warn-and-fix this module refuses. The typed
    member is what the construct index records, so a blocked map says which one it is.
    """
    text = _GENBANK.replace(
        "     gene            join(18..20,1..2)\n", f"     gene            {location}\n"
    )
    path = tmp_path / "pBAD.gb"
    path.write_text(text)
    with pytest.raises(GenBankLocationError) as caught:
        parse_genbank_component(str(path), "k")
    assert caught.value.refusal is refusal
    assert "pBAD.gb" in str(caught.value)


def test_a_wrap_on_a_linear_replicon_is_refused_rather_than_reinterpreted(
    tmp_path: Path,
) -> None:
    """A linear map whose feature crosses the origin contradicts its own topology line.

    Two statements of the same file disagree and neither is privileged, so the file is
    reported rather than read. The COMPACT form on a linear record never reaches that
    check, because Biopython applies its origin-wrap rewrite only to circular records
    and otherwise sets the location to ``None``; that case is an unparsable location,
    which is the honest reading of "Biopython could not read it".
    """
    linear = _GENBANK.replace("circular SYN", "linear   SYN")
    explicit = tmp_path / "pLINJOIN.gb"
    explicit.write_text(linear)
    with pytest.raises(GenBankLocationError) as caught:
        parse_genbank_component(str(explicit), "k")
    assert caught.value.refusal is LocationRefusal.origin_wrap_on_linear_topology
    assert "no origin to cross" in str(caught.value)
    compact = tmp_path / "pLINWRAP.gb"
    compact.write_text(
        linear.replace(
            "     gene            join(18..20,1..2)\n", "     gene            18..2\n"
        )
    )
    with pytest.raises(GenBankLocationError) as second:
        parse_genbank_component(str(compact), "k")
    assert second.value.refusal is LocationRefusal.unparsable_location


def test_the_retired_refusals_are_never_produced_by_a_parse(tmp_path: Path) -> None:
    """The two members the hardening used are kept only so an older index still loads.

    ``origin_wrapping_location`` and ``compound_location`` were every one of the 11
    Cachera refusals; both cases are now held, so no parse can emit them. They stay in
    the vocabulary because the index deposited on 2026-10-15 recorded them, and a
    vocabulary a stored record uses is not removed for tidiness.
    """
    held = {}
    for location in ("18..2", "join(18..20,1..2)", "join(3..6,10..14)"):
        text = _GENBANK.replace(
            "     gene            join(18..20,1..2)\n",
            f"     gene            {location}\n",
        )
        path = tmp_path / "pOK.gb"
        path.write_text(text)
        wrap = parse_genbank_component(str(path), "k").get_feature("wrap")
        held[location] = (wrap.composition, len(wrap.locations))
    assert held == {
        "18..2": (SpanComposition.origin_spanning, 2),
        "join(18..20,1..2)": (SpanComposition.origin_spanning, 2),
        "join(3..6,10..14)": (SpanComposition.joined, 2),
    }
    assert RETIRED_LOCATION_REFUSALS == {
        LocationRefusal.origin_wrapping_location,
        LocationRefusal.compound_location,
    }


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
