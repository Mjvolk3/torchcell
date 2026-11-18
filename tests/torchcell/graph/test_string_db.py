# tests/torchcell/graph/test_string_db.py
# [[tests.torchcell.graph.test_string_db]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/graph/test_string_db.py
"""``StringTaxonMirror`` over a six-protein synthetic slice under ``tmp_path``.

Nothing here reaches the network. :func:`_mirror` writes the three gzipped files
STRING publishes for one organism into the mirror layout and calls
:meth:`StringTaxonMirror.fetch`, which pins a file that is present but unpinned
without downloading it. That is the same code path a real mirror takes on its
second call, so the pinning behavior is exercised for real.

The synthetic links table, taxon 9999, with one row per direction as STRING
publishes them:

===========  ===========  ============  ======  =============
protein1     protein2     neighborhood  fusion  experimental
===========  ===========  ============  ======  =============
G_0001       G_0002       400           0       700
G_0001       G_0003       0             250     0
G_0002       G_0003       100           0       400
G_9999       G_0001       900           0       900
===========  ===========  ============  ======  =============

``G_9999`` is absent from the gene set handed to the builders, so every
assertion about it pins the gene-set filter.
"""

import gzip
from pathlib import Path

import pytest

from torchcell.graph.string_db import (
    STRING_SCORE_CHANNELS,
    MirrorDriftError,
    StringAsset,
    StringTaxonMirror,
    build_string_gene_multigraph,
    string_channel_graphs,
    string_graph_name,
)
from torchcell.sequence import GeneSet

TAXON = 9999
GENE_SET = GeneSet({"G_0001", "G_0002", "G_0003"})

_LINKS_HEADER = (
    "protein1 protein2 neighborhood fusion cooccurence coexpression "
    "experimental database textmining combined_score"
)
_LINK_ROWS = [
    ("G_0001", "G_0002", 400, 0, 0, 0, 700, 0, 0, 800),
    ("G_0001", "G_0003", 0, 250, 0, 0, 0, 0, 0, 250),
    ("G_0002", "G_0003", 100, 0, 0, 0, 400, 0, 0, 450),
    ("G_9999", "G_0001", 900, 0, 0, 0, 900, 0, 0, 950),
]

_INFO = "\n".join(
    ["#string_protein_id\tpreferred_name\tprotein_size\tannotation"]
    + [f"{TAXON}.G_000{i}\tgene{i}\t100\tannotation {i}" for i in (1, 2, 3)]
)

_ALIASES = "\n".join(
    ["#string_protein_id\talias\tsource"]
    + [f"{TAXON}.G_000{i}\tG_000{i}\tRefSeq_locus" for i in (1, 2, 3)]
    + [f"{TAXON}.G_000{i}\tACC0000{i}.1\tRefSeq" for i in (1, 2, 3)]
    + [f"{TAXON}.G_000{i}\tG_000{i}\tRefSeq" for i in (1, 2, 3)]
)


def _links_text() -> str:
    """The links table, each pair written in both directions as STRING does."""
    text = [_LINKS_HEADER]
    for row in _LINK_ROWS:
        a, b, *scores = row
        text.append(f"{TAXON}.{a} {TAXON}.{b} " + " ".join(str(v) for v in scores))
        text.append(f"{TAXON}.{b} {TAXON}.{a} " + " ".join(str(v) for v in scores))
    return "\n".join(text) + "\n"


def _mirror(tmp_path: Path) -> StringTaxonMirror:
    mirror = StringTaxonMirror(root=str(tmp_path / "string"), taxon=TAXON)
    Path(mirror.dir).mkdir(parents=True, exist_ok=True)
    payloads = {
        StringAsset.LINKS_DETAILED: _links_text(),
        StringAsset.INFO: _INFO + "\n",
        StringAsset.ALIASES: _ALIASES + "\n",
    }
    for asset, text in payloads.items():
        with gzip.open(mirror.path(asset), "wt") as handle:
            handle.write(text)
    mirror.fetch(assets=tuple(payloads))
    return mirror


def test_unsupported_release_is_refused_by_name() -> None:
    """A release this module does not serve raises at construction."""
    with pytest.raises(ValueError, match="unsupported STRING release"):
        StringTaxonMirror(root="/tmp", taxon=TAXON, release="11.0")


def test_filename_and_url_carry_the_taxon_and_release(tmp_path: Path) -> None:
    """The filename and URL are built from the taxon, asset and release."""
    mirror = StringTaxonMirror(root=str(tmp_path), taxon=160488, release="12.5")
    assert (
        mirror.filename(StringAsset.LINKS_DETAILED)
        == "160488.protein.links.detailed.v12.5.txt.gz"
    )
    assert mirror.url(StringAsset.LINKS_DETAILED) == (
        "https://stringdb-downloads.org/download/protein.links.detailed.v12.5/"
        "160488.protein.links.detailed.v12.5.txt.gz"
    )
    assert mirror.filename(StringAsset.SEQUENCES).endswith(".fa.gz")
    assert mirror.dir.endswith("v12.5/160488")


def test_strip_prefix_only_removes_this_taxon(tmp_path: Path) -> None:
    """Another taxon's prefix is left alone rather than sliced blindly."""
    mirror = StringTaxonMirror(root=str(tmp_path), taxon=160488)
    assert mirror.strip_prefix("160488.PP_0001") == "PP_0001"
    assert mirror.strip_prefix("4932.YAL001C") == "4932.YAL001C"


def test_fetch_pins_present_files_and_verifies(tmp_path: Path) -> None:
    """A present but unpinned file is pinned, and verify confirms it."""
    mirror = _mirror(tmp_path)
    manifest = mirror.manifest()
    assert manifest.taxon == TAXON
    assert manifest.release == "12.5"
    assert {record.asset for record in manifest.files} == {
        "protein.links.detailed",
        "protein.info",
        "protein.aliases",
    }
    assert manifest.upstream_checksums is False
    assert mirror.verify() == {record.path: True for record in manifest.files}


def test_fetch_refuses_to_repin_drifted_bytes(tmp_path: Path) -> None:
    """Editing a mirrored file makes fetch raise rather than bless it."""
    mirror = _mirror(tmp_path)
    with gzip.open(mirror.path(StringAsset.INFO), "wt") as handle:
        handle.write(_INFO + "\n#tampered\n")
    assert mirror.verify()[mirror.filename(StringAsset.INFO)] is False
    with pytest.raises(MirrorDriftError, match="refusing to re-pin"):
        mirror.fetch(assets=(StringAsset.INFO,))


def test_fetch_keeps_records_for_assets_it_was_not_asked_for(tmp_path: Path) -> None:
    """A narrower fetch does not delete the other files' provenance."""
    mirror = _mirror(tmp_path)
    mirror.fetch(assets=(StringAsset.INFO,))
    assert len(mirror.manifest().files) == 3


def test_fetch_stamps_last_check_without_moving_retrieved_at(tmp_path: Path) -> None:
    """Nothing was retrieved, so only last_check changes."""
    mirror = _mirror(tmp_path)
    before = {r.path: r.retrieval for r in mirror.manifest().files}
    mirror.fetch(assets=(StringAsset.INFO,))
    after = {r.path: r.retrieval for r in mirror.manifest().files}
    name = mirror.filename(StringAsset.INFO)
    assert after[name].retrieved_at == before[name].retrieved_at
    assert after[name].last_check is not None
    assert before[name].last_check is None


def test_verify_reports_a_missing_file_as_false(tmp_path: Path) -> None:
    """A deleted file is reported, not raised on."""
    mirror = _mirror(tmp_path)
    Path(mirror.path(StringAsset.INFO)).unlink()
    assert mirror.verify()[mirror.filename(StringAsset.INFO)] is False


def test_reads_verify_against_the_manifest(tmp_path: Path) -> None:
    """A read of drifted bytes raises instead of returning them."""
    mirror = _mirror(tmp_path)
    assert len(mirror.links_detailed()) == 8
    with gzip.open(mirror.path(StringAsset.LINKS_DETAILED), "wt") as handle:
        handle.write(_links_text() + f"{TAXON}.G_0003 {TAXON}.G_0001 1 0 0 0 0 0 0 1\n")
    fresh = StringTaxonMirror(root=mirror.root, taxon=TAXON)
    with pytest.raises(MirrorDriftError, match="hashes to"):
        fresh.links_detailed()


def test_links_detailed_strips_the_taxon_prefix(tmp_path: Path) -> None:
    """Both id columns come back as bare protein ids."""
    frame = _mirror(tmp_path).links_detailed()
    assert set(frame["protein1"]) == {"G_0001", "G_0002", "G_0003", "G_9999"}
    assert frame.loc[
        (frame.protein1 == "G_0001") & (frame.protein2 == "G_0002"), "experimental"
    ].tolist() == [700]


def test_preferred_names_maps_every_protein(tmp_path: Path) -> None:
    """The info table becomes an id to display-name map."""
    assert _mirror(tmp_path).preferred_names() == {
        "G_0001": "gene1",
        "G_0002": "gene2",
        "G_0003": "gene3",
    }


def test_alias_sources_reports_the_real_source_names(tmp_path: Path) -> None:
    """The available sources and their row counts are published."""
    assert _mirror(tmp_path).alias_sources() == {"RefSeq": 6, "RefSeq_locus": 3}


def test_alias_map_refuses_an_unknown_source(tmp_path: Path) -> None:
    """A source absent from the table raises rather than returning nothing."""
    with pytest.raises(ValueError, match="are not in the v12.5 table"):
        _mirror(tmp_path).alias_map(("NotASource",))


def test_alias_map_refuses_an_ambiguous_source(tmp_path: Path) -> None:
    """Two aliases for one protein in one source would let file order decide."""
    with pytest.raises(ValueError, match="more than one alias"):
        _mirror(tmp_path).alias_map(("RefSeq",))


def test_alias_map_returns_an_unambiguous_source(tmp_path: Path) -> None:
    """A one-alias-per-protein source maps cleanly."""
    assert _mirror(tmp_path).alias_map(("RefSeq_locus",)) == {
        "G_0001": "G_0001",
        "G_0002": "G_0002",
        "G_0003": "G_0003",
    }


def test_channel_graphs_filter_to_the_gene_set_and_carry_the_score(
    tmp_path: Path,
) -> None:
    """G_9999 is dropped, and each edge weight is that channel's score."""
    graphs = string_channel_graphs(_mirror(tmp_path), GENE_SET)
    assert set(graphs) == set(STRING_SCORE_CHANNELS)
    neighborhood = graphs["neighborhood"]
    assert sorted(neighborhood.nodes) == ["G_0001", "G_0002", "G_0003"]
    assert sorted(tuple(sorted(e)) for e in neighborhood.edges) == [
        ("G_0001", "G_0002"),
        ("G_0002", "G_0003"),
    ]
    assert neighborhood.edges["G_0001", "G_0002"]["weight"] == 400
    assert neighborhood.edges["G_0001", "G_0002"]["version"] == "12.5"
    assert graphs["fusion"].number_of_edges() == 1
    assert graphs["fusion"].edges["G_0001", "G_0003"]["weight"] == 250
    assert graphs["cooccurence"].number_of_edges() == 0


def test_above_score_is_a_strict_threshold(tmp_path: Path) -> None:
    """A score equal to the threshold is excluded, which is what the name says."""
    mirror = _mirror(tmp_path)
    assert (
        string_channel_graphs(mirror, GENE_SET, channels=("neighborhood",))[
            "neighborhood"
        ].number_of_edges()
        == 2
    )
    at_400 = string_channel_graphs(
        mirror, GENE_SET, channels=("neighborhood",), above_score=400
    )["neighborhood"]
    assert at_400.number_of_edges() == 0
    at_399 = string_channel_graphs(
        mirror, GENE_SET, channels=("neighborhood",), above_score=399
    )["neighborhood"]
    assert at_399.number_of_edges() == 1


def test_unknown_channel_is_refused(tmp_path: Path) -> None:
    """A misspelled channel raises and lists the valid ones."""
    with pytest.raises(ValueError, match="unknown STRING channel"):
        string_channel_graphs(_mirror(tmp_path), GENE_SET, channels=("cooccurrence",))


def test_incomplete_protein_to_gene_is_refused(tmp_path: Path) -> None:
    """A partial rename would drop edges silently, so it raises."""
    with pytest.raises(ValueError, match="are unmapped"):
        string_channel_graphs(
            _mirror(tmp_path), GENE_SET, protein_to_gene={"G_0001": "a"}
        )


def test_graph_name_matches_the_yeast_form() -> None:
    """The graph name is the release with dots replaced, then the channel."""
    assert string_graph_name("12.5", "neighborhood") == "string12_5_neighborhood"
    assert string_graph_name("12.0", "fusion") == "string12_0_fusion"


def test_multigraph_is_keyed_by_graph_name(tmp_path: Path) -> None:
    """The multigraph carries one named GeneGraph per channel."""
    multigraph = build_string_gene_multigraph(_mirror(tmp_path), GENE_SET)
    assert sorted(multigraph.keys()) == [
        string_graph_name("12.5", channel) for channel in sorted(STRING_SCORE_CHANNELS)
    ]
    graph = multigraph["string12_5_neighborhood"]
    assert graph.name == "string12_5_neighborhood"
    assert graph.graph.number_of_edges() == 2
    assert set(graph.max_gene_set) == set(GENE_SET)
