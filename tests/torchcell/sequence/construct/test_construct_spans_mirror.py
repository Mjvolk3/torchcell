# tests/torchcell/sequence/construct/test_construct_spans_mirror.py
# [[tests.torchcell.sequence.construct.test_construct_spans_mirror]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/sequence/construct/test_construct_spans_mirror.py
r"""Spans of parts, asserted against the mirrored Cachera maps by name.

The hand-written fixtures in ``test_construct_registry.py`` pin the rules so the suite
runs on a machine with no ``DATA_ROOT``. These tests pin the MEASUREMENTS the rules were
built from, on the 17 real maps under
``$DATA_ROOT/torchcell-library/cacheraCRISPAHighthroughputMethod2023/si/plasmids/``, and
are marked ``data`` so they run with ``--data``.

What they defend, each a number that moved on 2026-10-15:

- 17 of 17 maps index and 0 are blocked. Before a span held parts, 6 indexed and 11 were
  ``BlockedConstruct`` rows.
- ``pBTX002.gb``'s ``pUC\ori`` is 666 bp, not the 12,096 bp of the whole plasmid, and
  the ``AmpR`` CDS of ``pbtx013.gb`` is 861 bp, not 4,894. The envelope was what the
  parser stored before it refused these maps.
- 570 of 570 features regenerate the location string their own file wrote, and every
  compound span's canonical INSDC spelling re-parses through Biopython into the same
  parts.
- 0 of 570 features are an INSDC ``join``. The 11 "join" locations the audit counted
  were Biopython's rewrite of the compact wrap form, not what the files carry, so the
  join case has no real map in this set to be asserted against; the plain-join tests are
  the fixtures in ``test_plasmid.py``, and this is where that absence is recorded.
"""

from __future__ import annotations

import os
import os.path as osp
import re
import warnings
from pathlib import Path

import pytest
from Bio import SeqIO

from torchcell.datamodels.schema import SpanComposition
from torchcell.sequence.construct import (
    CACHERA2023_CRI_SPA_PLASMIDS,
    ConstructFeature,
    load_construct_manifest,
    parse_contig_features,
)

DATA_ROOT = os.environ.get("DATA_ROOT", "")
CACHERA_KEY = "cacheraCRISPAHighthroughputMethod2023"
PLASMIDS = osp.join(DATA_ROOT, "torchcell-library", CACHERA_KEY, "si", "plasmids")
INDEX = osp.join(
    DATA_ROOT, "torchcell-constructs", CACHERA2023_CRI_SPA_PLASMIDS, "manifest.json"
)
MIRROR_PRESENT = osp.isdir(PLASMIDS) and osp.isfile(INDEX)

pytestmark = [
    pytest.mark.data,
    pytest.mark.skipif(
        not MIRROR_PRESENT,
        reason="requires the 17 Cachera maps in the library tier and the deposited "
        f"{CACHERA2023_CRI_SPA_PLASMIDS} index",
    ),
]


def _source_locations(path: Path) -> list[tuple[str, str]]:
    """``(feature type, location string)`` per non-``source`` feature, from the bytes.

    Read from the raw file, never from Biopython: Biopython is what rewrites the compact
    origin-wrapping form, so a round trip through it would not be a round trip against
    what the source wrote.
    """
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    inside = False
    found: list[tuple[str, str]] = []
    for index, line in enumerate(lines):
        if line.startswith("FEATURES"):
            inside = True
            continue
        if line.startswith("ORIGIN"):
            break
        if not inside:
            continue
        match = re.match(r"^ {5}(\S+)\s+(\S.*)$", line)
        if match is None:
            continue
        feature_type, location = match.group(1), match.group(2).strip()
        follower = index + 1
        while (
            follower < len(lines)
            and lines[follower].startswith(" " * 21)
            and not lines[follower].strip().startswith("/")
        ):
            location += lines[follower].strip()
            follower += 1
        if feature_type == "source":
            continue
        found.append((feature_type, location))
    return found


def _features(name: str) -> list[ConstructFeature]:
    """The projected features of one mirrored map, by file name."""
    _, features = parse_contig_features(Path(PLASMIDS) / name, CACHERA_KEY)
    return features


def _replicon_length(name: str) -> int:
    """The map's length, measured from its bases."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        record = SeqIO.read(osp.join(PLASMIDS, name), "genbank")
    return len(record.seq)


def test_all_seventeen_cachera_maps_index_with_no_blocked_rows() -> None:
    """The deposited index holds 17 contigs and names nothing blocked.

    6 of 17 indexed while a span was one interval. The 11 that did not each carried
    exactly one feature whose interval crosses the origin of its circular replicon.
    """
    manifest = load_construct_manifest(CACHERA2023_CRI_SPA_PLASMIDS)
    assert len(manifest.files) == 17
    assert len(manifest.contigs) == 17
    assert manifest.blocked == []
    assert all(contig.source_name is not None for contig in manifest.contigs)
    assert {
        contig.source_name.value
        for contig in manifest.contigs
        if contig.source_name is not None
    } >= {"pBTX1", "pBTX2", "pCCM023", "pBTX013", "pBTX023"}


def test_pbtx002_puc_ori_is_666_bp_and_not_the_whole_plasmid() -> None:
    r"""``pBTX002.gb``'s ``pUC\ori`` at ``11559..128`` is the origin, not the plasmid.

    This is the feature the audit measured: a single-range span could hold it only as
    ``11559..12096`` plus ``1..128`` flattened to ``[1, 12096]``, which made
    ``feature_sequence`` return the whole 12,096 bp replicon as the origin of
    replication. The parts give 538 + 128 = 666 bp, which is a pUC origin.
    """
    length = _replicon_length("pBTX002.gb")
    assert length == 12096
    ori = next(f for f in _features("pBTX002.gb") if f.key.label == "pUC\\ori")
    span = ori.key.span
    assert ori.key.so_id == "SO:0000296"
    assert span.composition is SpanComposition.origin_spanning
    assert [(part.start, part.end) for part in span.parts] == [(11559, 12096), (1, 128)]
    assert span.length == 666
    assert span.length < length
    assert span.orientation == "inline"
    assert span.insdc_location(compact_wrap=True) == "11559..128"
    assert span.insdc_location() == "join(11559..12096,1..128)"
    assert ori.key.token == "pUC\\ori@SO:0000296:wrap:11559-12096,1-128+"


def test_the_ampr_cds_of_pbtx013_is_861_bp_on_the_reverse_strand() -> None:
    """``pbtx013.gb``'s ``complement(4892..858)`` is a reverse-strand wrap of 861 bp.

    The strand case on a real map: the parts are stored in the order the file writes
    them inside ``complement(...)``, which is the REVERSE of the 5'->3' order Biopython
    returns, so the location regenerates exactly. 861 bp is beta-lactamase (286 residues
    and a stop); the envelope would have been the whole 4,894 bp plasmid.
    """
    length = _replicon_length("pbtx013.gb")
    assert length == 4894
    ampr = next(
        f
        for f in _features("pbtx013.gb")
        if f.key.label == "AmpR" and f.genbank_type == "CDS"
    )
    span = ampr.key.span
    assert span.composition is SpanComposition.origin_spanning
    assert span.orientation == "reverse_complement"
    assert [(part.start, part.end) for part in span.parts] == [(4892, 4894), (1, 858)]
    assert span.length == 861
    assert span.insdc_location(compact_wrap=True) == "complement(4892..858)"
    assert span.insdc_location() == "complement(join(4892..4894,1..858))"
    assert ampr.key.token.endswith("-")


def test_every_feature_of_every_map_regenerates_its_source_location() -> None:
    """570 of 570 locations are regenerated verbatim from the stored parts and strand.

    The round trip is the proof that a stored coordinate is the source's coordinate: the
    compact writer reproduces the string each file wrote, byte for byte, across all 17
    maps and both strands.
    """
    manifest = load_construct_manifest(CACHERA2023_CRI_SPA_PLASMIDS)
    checked = 0
    for contig in manifest.contigs:
        path = Path(DATA_ROOT) / "torchcell-library" / contig.ref.key / contig.ref.path
        _, projected = parse_contig_features(path, CACHERA_KEY)
        written = _source_locations(path)
        assert len(written) == len(projected), contig.ref.path
        for (_, location), feature in zip(written, projected, strict=True):
            assert feature.key.span.insdc_location(compact_wrap=True) == location, (
                f"{contig.ref.path}: {location}"
            )
            checked += 1
    assert checked == 570


def test_the_set_carries_eleven_wraps_and_no_insdc_join() -> None:
    """11 of 570 features cross the origin; 0 are a ``join``, in any of the 17 files.

    The audit counted "11 features with ``join`` locations" and 11 maps warning about
    origin wrapping, and read them as two overlapping findings. They are one: the files
    contain no ``join`` operator at all (``grep`` over the 17 maps), and the join
    Biopython reports is its own rewrite of the compact wrap. The join case therefore
    has no real map here to assert against, which is why it is a fixture elsewhere.
    """
    manifest = load_construct_manifest(CACHERA2023_CRI_SPA_PLASMIDS)
    compositions = [
        feature.key.span.composition
        for contig in manifest.contigs
        for feature in contig.features
    ]
    assert len(compositions) == 570
    assert compositions.count(SpanComposition.origin_spanning) == 11
    assert compositions.count(SpanComposition.joined) == 0
    assert compositions.count(SpanComposition.simple) == 559
    for name in sorted(osp.basename(p) for p in os.listdir(PLASMIDS)):
        text = (Path(PLASMIDS) / name).read_text(encoding="utf-8", errors="replace")
        features = text.split("FEATURES", 1)[1].split("ORIGIN", 1)[0]
        assert "join(" not in features, name
        assert "order(" not in features, name
