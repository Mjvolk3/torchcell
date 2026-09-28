# tests/torchcell/graph/test_graph.py
# [[tests.torchcell.graph.test_graph]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/graph/test_graph.py
"""``SCerevisiaeGraph`` on a stub genome under ``tmp_path``, plus the SGD-backed filters.

The stub genome has three genes, YAL001C, YAL002W, YAL003W, a ten-term GO DAG and an
alias map; the per-gene SGD JSON is written by hand into ``<sgd_root>/genes/``. Nothing
is downloaded: the STRING table is a three-row gzip and the TFLink table a three-row
TSV, both written by the test, and the download helpers are exercised only on their
"file already exists" and "unsupported version" branches.

Expected values, derived from the fixture:

* GO: YAL001C is annotated to GO:0000002 (child of GO:0000001, child of the BP root
  GO:0008150), GO:0000003 (child of the MF root GO:0003674) and the obsolete GO:0000009;
  YAL002W to GO:0000004 (child of the CC root GO:0005575) and GO:0000001; YAL003W to the
  three roots and to GO:0000011, whose only parent GO:0000010 annotates no gene. So
  ``all_go_terms`` is the eight non-obsolete annotated ids, ``create_G_go`` has those
  eight plus ``GO:ROOT``, GO:0000010 never enters the graph, and GO:0000011 is rewired
  straight to the BP root.
* STRING v12.0: rows YAL001C-YAL002W (coexpression 150, experimental 900),
  YAL001C-YZZ999W (neighborhood 300, dropped: not in the gene set), YAL002W-YAL003W
  (database 500).
* TFLink: TFA1 -> YAL001C, TGT1 -> [YAL003W, YAL002W], BAD -> [YZZ999W]; rows TFA1->TGT1
  (two edges), TFA1->BAD (dropped, target outside the gene set), NOPE->TGT1 (dropped,
  unknown TF).

The two SGD-backed filter tests at the end read the real genome and stay behind
``--data``; ``tests/torchcell/graph/test_gene_graph.py`` pins the filters on hand-built
graphs.
"""

import gzip
import json
import logging
import os
import os.path as osp
import pickle
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import networkx as nx
import pytest
from dotenv import load_dotenv
from tqdm import tqdm

from torchcell.graph import SCerevisiaeGraph, filter_by_date, filter_go_IGI
from torchcell.graph.graph import (
    SCEREVISIAE_GENE_GRAPH_MAP,
    SCEREVISIAE_GENE_GRAPH_VALID_NAMES,
    GeneGraph,
    build_gene_multigraph,
)
from torchcell.sequence import GeneSet, ParsedGenome
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome

load_dotenv()
DATA_ROOT = os.getenv("DATA_ROOT")

GENES = GeneSet(["YAL001C", "YAL002W", "YAL003W"])
STRING_FILE = "4932.protein.links.detailed.v12.0.txt.gz"
TFLINK_FILE = "TFLink_Saccharomyces_cerevisiae_interactions_All_simpleFormat_v1.0.tsv"


class _Term:
    """The GOTerm attributes ``create_go_subgraph`` and ``all_go_terms`` read."""

    def __init__(
        self,
        go_id: str,
        name: str,
        namespace: str,
        level: int,
        parents: list["_Term"] | None = None,
        obsolete: bool = False,
    ) -> None:
        self.id = go_id
        self.item_id = go_id
        self.name = name
        self.namespace = namespace
        self.level = level
        self.depth = level
        self.is_obsolete = obsolete
        self.alt_ids: set[str] = set()
        self.parents = set(parents or [])


def _go_dag() -> dict[str, _Term]:
    bp_root = _Term("GO:0008150", "biological_process", "biological_process", 0)
    bp_a = _Term("GO:0000001", "term a", "biological_process", 1, [bp_root])
    bp_b = _Term("GO:0000002", "term b", "biological_process", 2, [bp_a])
    bp_x = _Term("GO:0000010", "unannotated", "biological_process", 1, [bp_root])
    bp_c = _Term("GO:0000011", "term c", "biological_process", 2, [bp_x])
    mf_root = _Term("GO:0003674", "molecular_function", "molecular_function", 0)
    mf_a = _Term("GO:0000003", "term mf", "molecular_function", 1, [mf_root])
    cc_root = _Term("GO:0005575", "cellular_component", "cellular_component", 0)
    cc_a = _Term("GO:0000004", "term cc", "cellular_component", 1, [cc_root])
    gone = _Term("GO:0000009", "gone", "biological_process", 3, obsolete=True)
    terms = [bp_root, bp_a, bp_b, bp_x, bp_c, mf_root, mf_a, cc_root, cc_a, gone]
    return {term.id: term for term in terms}


class _StubGenome:
    """The genome attributes ``SCerevisiaeGraph`` reads: gene_set, go_dag, alias map."""

    def __init__(self, alias_to_systematic: dict[str, list[str]] | None) -> None:
        self.gene_set = GENES
        self.go_dag = _go_dag()
        if alias_to_systematic is not None:
            self.alias_to_systematic = alias_to_systematic


ALIASES = {"TFA1": ["YAL001C"], "TGT1": ["YAL003W", "YAL002W"], "BAD": ["YZZ999W"]}


def _detail(
    go_id: str, experiment: str = "IDA", date: str = "2015-01-01"
) -> dict[str, Any]:
    return {
        "go": {"go_id": go_id},
        "experiment": {"display_name": experiment},
        "date_created": date,
    }


def _raw_nodes() -> dict[str, dict[str, Any]]:
    return {
        "YAL001C": {
            "locus": {
                "protein_overview": {
                    "length": 100,
                    "molecular_weight": 11000.5,
                    "pi": 6.5,
                },
                "pathways": [{"pathway": {"display_name": "glycolysis"}}],
            },
            "sequence_details": {
                "genomic_dna": [
                    {
                        "strain": {"display_name": "S288C"},
                        "start": 10,
                        "end": 20,
                        "contig": {"display_name": "Chromosome I"},
                    }
                ]
            },
            "go_details": [
                _detail("GO:0000002"),
                _detail("GO:0000003"),
                _detail("GO:0000009"),
            ],
            "interaction_details": [
                {
                    "interaction_type": "Physical",
                    "locus1": {"format_name": "YAL001C"},
                    "locus2": {"format_name": "YAL002W"},
                    "note": "phys",
                },
                {
                    "interaction_type": "Genetic",
                    "locus1": {"format_name": "YAL001C"},
                    "locus2": {"format_name": "YAL003W"},
                    "note": "gen",
                },
                {
                    "interaction_type": "Physical",
                    "locus1": {"format_name": "YAL001C"},
                    "locus2": {"format_name": "YZZ999W"},
                },
            ],
            "regulation_details": [
                {
                    "locus1": {"format_name": "YAL001C", "display_name": "TFA1"},
                    "locus2": {"format_name": "YAL003W", "display_name": "TGT1"},
                    "regulation_type": "transcription",
                }
            ],
        },
        "YAL002W": {
            "locus": {"pathways": []},
            "sequence_details": {
                "genomic_dna": [
                    {
                        "strain": {"display_name": "W303"},
                        "start": 1,
                        "end": 2,
                        "contig": {"display_name": "x"},
                    }
                ]
            },
            "go_details": [_detail("GO:0000004"), _detail("GO:0000001")],
        },
        "YAL003W": {
            "locus": {"pathways": []},
            "sequence_details": {"genomic_dna": []},
            "go_details": [
                _detail("GO:0008150", "ND", "2001-01-01"),
                _detail("GO:0003674", "ND", "2001-01-01"),
                _detail("GO:0005575", "ND", "2001-01-01"),
                _detail("GO:0000011", "IGI", "2020-05-05"),
            ],
        },
    }


def _make_graph(root: Path, genome: _StubGenome) -> SCerevisiaeGraph:
    genes_dir = root / "sgd" / "genes"
    genes_dir.mkdir(parents=True, exist_ok=True)
    for gene, data in _raw_nodes().items():
        (genes_dir / f"{gene}.json").write_text(json.dumps(data), encoding="utf-8")
    return SCerevisiaeGraph(
        sgd_root=str(root / "sgd"),
        string_root=str(root / "string"),
        tflink_root=str(root / "tflink"),
        genome=genome,  # type: ignore[arg-type]  # duck-typed stub
    )


@pytest.fixture
def sgd_graph(tmp_path: Path) -> SCerevisiaeGraph:
    """A graph builder over the stub genome with the alias map."""
    return _make_graph(tmp_path, _StubGenome(ALIASES))


def _write_string_table(graph: SCerevisiaeGraph) -> str:
    version_dir = osp.join(graph.string_root, "v12.0")
    os.makedirs(version_dir, exist_ok=True)
    path = osp.join(version_dir, STRING_FILE)
    with gzip.open(path, "wt") as handle:
        handle.write(
            "protein1 protein2 neighborhood fusion cooccurence coexpression "
            "experimental database textmining combined_score\n"
            "4932.YAL001C 4932.YAL002W 0 0 0 150 900 0 0 910\n"
            "4932.YAL001C 4932.YZZ999W 300 0 0 0 0 0 0 300\n"
            "4932.YAL002W 4932.YAL003W 0 0 0 0 0 500 0 500\n"
        )
    return path


def _write_tflink_table(graph: SCerevisiaeGraph, rows: list[tuple[str, str]]) -> str:
    path = osp.join(graph.tflink_root, TFLINK_FILE)
    lines = ["UniprotID.TF\tUniprotID.Target\tName.TF\tName.Target\tDetection.method"]
    lines += [f"P1\tP2\t{tf}\t{target}\tchip" for tf, target in rows]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_post_init_names_one_json_per_gene_and_creates_the_graph_directories(
    tmp_path: Path, sgd_graph: SCerevisiaeGraph
) -> None:
    """``json_files`` follows the sorted gene set; each root gets a ``graph/`` directory."""
    assert sgd_graph.json_files == ["YAL001C.json", "YAL002W.json", "YAL003W.json"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["sgd", "string", "tflink"]
    assert sorted(p.name for p in (tmp_path / "sgd").iterdir()) == ["genes", "graph"]
    assert [p.name for p in (tmp_path / "string").iterdir()] == ["graph"]
    assert [p.name for p in (tmp_path / "tflink").iterdir()] == ["graph"]


def test_g_raw_loads_each_gene_json_as_a_node_and_caches_a_pickle(
    tmp_path: Path, sgd_graph: SCerevisiaeGraph
) -> None:
    """Node = file stem, attributes = the JSON; ``G_raw.pkl`` is written on first access."""
    raw = sgd_graph.G_raw
    assert sorted(raw.nodes) == ["YAL001C", "YAL002W", "YAL003W"]
    assert raw.nodes["YAL001C"]["locus"]["pathways"] == [
        {"pathway": {"display_name": "glycolysis"}}
    ]
    assert raw.nodes["YAL003W"] == _raw_nodes()["YAL003W"]
    with open(tmp_path / "sgd" / "graph" / "G_raw.pkl", "rb") as handle:
        assert sorted(pickle.load(handle).nodes) == sorted(raw.nodes)


def test_g_raw_prefers_an_existing_pickle_over_the_json_files(tmp_path: Path) -> None:
    """A ``G_raw.pkl`` already on disk is what ``G_raw`` returns; the JSON is not read."""
    graph = _make_graph(tmp_path, _StubGenome(ALIASES))
    cached = nx.Graph()
    cached.add_node("CACHED", locus={})
    graph.save_graph(cached, "G_raw", root_type="sgd")
    assert list(graph.G_raw.nodes) == ["CACHED"]


def test_all_go_terms_skips_obsolete_ids_and_go_to_genes_maps_each_term_to_its_genes(
    sgd_graph: SCerevisiaeGraph,
) -> None:
    """GO:0000009 is obsolete and excluded; every other annotated id maps to its genes."""
    assert list(sgd_graph.all_go_terms) == [
        "GO:0000001",
        "GO:0000002",
        "GO:0000003",
        "GO:0000004",
        "GO:0000011",
        "GO:0003674",
        "GO:0005575",
        "GO:0008150",
    ]
    assert {k: list(v) for k, v in sgd_graph.go_to_genes.items()} == {
        "GO:0000001": ["YAL002W"],
        "GO:0000002": ["YAL001C"],
        "GO:0000003": ["YAL001C"],
        "GO:0000004": ["YAL002W"],
        "GO:0000011": ["YAL003W"],
        "GO:0003674": ["YAL003W"],
        "GO:0005575": ["YAL003W"],
        "GO:0008150": ["YAL003W"],
    }


def test_create_go_subgraph_adds_child_to_parent_edges_and_rewires_orphans_to_the_root(
    sgd_graph: SCerevisiaeGraph,
) -> None:
    """Edges are child -> parent (graph.py:1082); a term whose parent is not in the
    term list loses that edge and is linked straight to the root.
    """
    sub = sgd_graph.create_go_subgraph(
        ["GO:0000002", "GO:0000001", "GO:0008150", "GO:0000011"],
        sgd_graph.genome.go_dag,
    )
    assert sorted(sub.nodes) == ["GO:0000001", "GO:0000002", "GO:0000011", "GO:0008150"]
    assert sorted(sub.edges) == [
        ("GO:0000001", "GO:0008150"),
        ("GO:0000002", "GO:0000001"),
        ("GO:0000011", "GO:0008150"),
    ]
    node = dict(sub.nodes["GO:0000002"])
    assert list(node.pop("gene_set")) == ["YAL001C"]
    assert node == {
        "id": "GO:0000002",
        "item_id": "GO:0000002",
        "name": "term b",
        "namespace": "biological_process",
        "level": 2,
        "depth": 2,
        "is_obsolete": False,
        "alt_ids": set(),
        "genes": {"YAL001C": {"go_details": _detail("GO:0000002")}},
    }


def test_create_go_subgraph_raises_on_a_term_that_annotates_no_gene(
    sgd_graph: SCerevisiaeGraph,
) -> None:
    """Finding: ``gene_set`` is ``go_to_genes.get(go_id, None)`` and line 1073 iterates
    it, so a term with no annotated genes raises TypeError instead of an empty node.
    ``create_G_go`` never hits this because it passes only annotated terms.
    """
    with pytest.raises(TypeError, match="'NoneType' object is not iterable"):
        sgd_graph.create_go_subgraph(["GO:0000010"], sgd_graph.genome.go_dag)


def test_create_go_subgraph_requires_exactly_one_level_zero_term(
    sgd_graph: SCerevisiaeGraph,
) -> None:
    """Without the root in the term list its data-less node is dropped and the assertion fires."""
    with pytest.raises(
        AssertionError, match="There should be only one root node for a GO subgraph"
    ):
        sgd_graph.create_go_subgraph(
            ["GO:0000002", "GO:0000001"], sgd_graph.genome.go_dag
        )


def test_combine_with_super_node_links_every_level_zero_node_under_go_root() -> None:
    """The super node has level -1; each level-0 node gets an edge to it."""
    a = nx.DiGraph()
    a.add_node("A", level=0)
    a.add_node("A1", level=1)
    a.add_edge("A1", "A")
    b = nx.DiGraph()
    b.add_node("B", level=0)
    combined = SCerevisiaeGraph.combine_with_super_node([a, b])
    assert sorted(combined.nodes(data=True)) == [
        ("A", {"level": 0}),
        ("A1", {"level": 1}),
        ("B", {"level": 0}),
        ("GO:ROOT", {"name": "GO Super Node", "namespace": "super_root", "level": -1}),
    ]
    assert sorted(combined.edges) == [("A", "GO:ROOT"), ("A1", "A"), ("B", "GO:ROOT")]


def test_g_go_joins_the_three_namespace_subgraphs_under_the_super_root(
    tmp_path: Path, sgd_graph: SCerevisiaeGraph
) -> None:
    """Eight annotated terms plus GO:ROOT; GO:0000010 (unannotated) never appears."""
    go = sgd_graph.G_go
    assert sorted(go.nodes) == [
        "GO:0000001",
        "GO:0000002",
        "GO:0000003",
        "GO:0000004",
        "GO:0000011",
        "GO:0003674",
        "GO:0005575",
        "GO:0008150",
        "GO:ROOT",
    ]
    assert sorted(go.edges) == [
        ("GO:0000001", "GO:0008150"),
        ("GO:0000002", "GO:0000001"),
        ("GO:0000003", "GO:0003674"),
        ("GO:0000004", "GO:0005575"),
        ("GO:0000011", "GO:0008150"),
        ("GO:0003674", "GO:ROOT"),
        ("GO:0005575", "GO:ROOT"),
        ("GO:0008150", "GO:ROOT"),
    ]
    assert (tmp_path / "sgd" / "graph" / "G_go.pkl").exists()


def test_g_gene_carries_protein_overview_loci_and_pathway_attributes(
    sgd_graph: SCerevisiaeGraph,
) -> None:
    """Only YAL001C has an S288C locus and a protein overview; pathways ``[]`` becomes None."""
    gene = sgd_graph.G_gene
    assert gene.name == "gene"
    empty = {
        "length": None,
        "molecular_weight": None,
        "pi": None,
        "median_value": None,
        "median_abs_dev_value": None,
        "pathways": None,
    }
    assert dict(gene.graph.nodes(data=True)) == {
        "YAL001C": {
            "length": 100,
            "molecular_weight": 11000.5,
            "pi": 6.5,
            "median_value": None,
            "median_abs_dev_value": None,
            "start": 10,
            "end": 20,
            "chromosome": "Chromosome I",
            "pathways": ["glycolysis"],
        },
        "YAL002W": empty,
        "YAL003W": empty,
    }


def test_physical_genetic_and_regulatory_graphs_come_from_the_raw_interaction_details(
    sgd_graph: SCerevisiaeGraph,
) -> None:
    """Physical/genetic edges keep the interaction dict; a partner outside G_raw is dropped;
    regulatory is directed and carries the locus dicts as node attributes.
    """
    raw = _raw_nodes()["YAL001C"]
    physical = sgd_graph.G_physical
    assert physical.name == "physical"
    assert list(physical.graph.edges(data=True)) == [
        ("YAL001C", "YAL002W", raw["interaction_details"][0])
    ]
    genetic = sgd_graph.G_genetic
    assert genetic.name == "genetic"
    assert list(genetic.graph.edges(data=True)) == [
        ("YAL001C", "YAL003W", raw["interaction_details"][1])
    ]
    regulatory = sgd_graph.G_regulatory
    assert regulatory.name == "regulatory"
    assert type(regulatory.graph) is nx.DiGraph
    assert list(regulatory.graph.edges(data=True)) == [
        ("YAL001C", "YAL003W", raw["regulation_details"][0])
    ]
    assert dict(regulatory.graph.nodes(data=True)) == {
        "YAL001C": {"format_name": "YAL001C", "display_name": "TFA1"},
        "YAL003W": {"format_name": "YAL003W", "display_name": "TGT1"},
    }


def test_create_string_graphs_strips_the_taxon_prefix_and_keeps_genome_genes_only(
    tmp_path: Path, sgd_graph: SCerevisiaeGraph
) -> None:
    """One channel graph per network type, weight = the channel score, version stamped;
    the YZZ999W row is dropped so neighborhood is empty; six pickles are written.
    """
    _write_string_table(sgd_graph)
    experimental = sgd_graph.G_string12_0_experimental
    assert experimental.name == "string12_0_experimental"
    assert list(experimental.graph.edges(data=True)) == [
        ("YAL001C", "YAL002W", {"weight": 900, "version": "12.0"})
    ]
    assert list(sgd_graph.G_string12_0_coexpression.graph.edges(data=True)) == [
        ("YAL001C", "YAL002W", {"weight": 150, "version": "12.0"})
    ]
    assert list(sgd_graph.G_string12_0_database.graph.edges(data=True)) == [
        ("YAL002W", "YAL003W", {"weight": 500, "version": "12.0"})
    ]
    assert list(sgd_graph.G_string12_0_neighborhood.graph.nodes) == []
    assert sorted(p.name for p in (tmp_path / "string" / "graph").iterdir()) == [
        "G_string12_0_coexpression.pkl",
        "G_string12_0_cooccurence.pkl",
        "G_string12_0_database.pkl",
        "G_string12_0_experimental.pkl",
        "G_string12_0_fusion.pkl",
        "G_string12_0_neighborhood.pkl",
    ]


def test_lazy_properties_load_a_pickle_already_in_the_graph_directory(
    sgd_graph: SCerevisiaeGraph,
) -> None:
    """A saved ``G_<name>.pkl`` short-circuits the build for sgd, string, and tflink roots."""
    toy = GeneGraph(name="toy", graph=nx.Graph(), max_gene_set=GENES)
    sgd_graph.save_graph(toy, "G_physical", root_type="sgd")
    sgd_graph.save_graph(toy, "G_string9_1_fusion", root_type="string")
    sgd_graph.save_graph(toy, "G_tflink", root_type="tflink")
    loaded = [sgd_graph.G_physical, sgd_graph.G_string9_1_fusion, sgd_graph.G_tflink]
    assert [g.name for g in loaded] == ["toy"] * 3
    assert [sorted(g.graph.nodes) for g in loaded] == [sorted(toy.graph.nodes)] * 3


def test_every_string_channel_property_loads_its_own_pickle(
    sgd_graph: SCerevisiaeGraph,
) -> None:
    """Each of the 18 ``string<v>_<channel>`` names reads ``G_string<v>_<channel>.pkl``."""
    names = [n for n in SCEREVISIAE_GENE_GRAPH_VALID_NAMES if n.startswith("string")]
    assert len(names) == 18
    saved = {n: GeneGraph(name=n, graph=nx.Graph(), max_gene_set=GENES) for n in names}
    for name, graph in saved.items():
        sgd_graph.save_graph(graph, f"G_{name}", root_type="string")
    assert [SCEREVISIAE_GENE_GRAPH_MAP[n](sgd_graph).name for n in names] == names


def test_create_g_tflink_maps_aliases_to_every_systematic_name_in_the_gene_set(
    sgd_graph: SCerevisiaeGraph,
) -> None:
    """TGT1 maps to two genes so the TFA1 row yields two edges; the BAD target (outside the
    gene set), the NOPE TF and the NOPE target (both unmapped) drop.
    """
    _write_tflink_table(
        sgd_graph,
        [("TFA1", "TGT1"), ("TFA1", "BAD"), ("NOPE", "TGT1"), ("TFA1", "NOPE")],
    )
    tflink = sgd_graph.G_tflink
    assert tflink.name == "tflink"
    assert type(tflink.graph) is nx.DiGraph
    row = {
        "UniprotID.TF": "P1",
        "UniprotID.Target": "P2",
        "Name.TF": "TFA1",
        "Name.Target": "TGT1",
        "Detection.method": "chip",
        "TF_systematic": "YAL001C",
    }
    assert sorted(tflink.graph.edges(data=True)) == [
        ("YAL001C", "YAL002W", {**row, "Target_systematic": "YAL002W"}),
        ("YAL001C", "YAL003W", {**row, "Target_systematic": "YAL003W"}),
    ]


def test_create_g_tflink_without_an_alias_map_accepts_only_systematic_names(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """No ``alias_to_systematic``: a name is kept only when it is itself in the gene set."""
    graph = _make_graph(tmp_path, _StubGenome(None))
    _write_tflink_table(graph, [("YAL001C", "YAL003W"), ("TFA1", "YAL003W")])
    with caplog.at_level(logging.WARNING, logger="torchcell.graph.graph"):
        tflink = graph.create_G_tflink()
    assert caplog.messages == [
        "Missing alias_to_systematic mapping in genome. TFLink functionality will be limited."
    ]
    assert list(tflink.edges) == [("YAL001C", "YAL003W")]
    assert tflink.edges["YAL001C", "YAL003W"]["Target_systematic"] == "YAL003W"


def test_parse_genome_returns_the_gene_set_and_writes_the_alias_map_onto_the_class(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Finding: ``setattr(ParsedGenome, ...)`` (graph.py:438) stores the alias map on the
    CLASS, so every ParsedGenome, including one parsed from a genome without an alias map,
    sees the last genome's mapping.
    """
    monkeypatch.setattr(ParsedGenome, "alias_to_systematic", None, raising=False)
    parsed = SCerevisiaeGraph.parse_genome(_StubGenome(ALIASES))
    assert list(parsed.gene_set) == ["YAL001C", "YAL002W", "YAL003W"]
    assert list(type(parsed).model_fields) == ["gene_set"]
    assert vars(ParsedGenome)["alias_to_systematic"] is ALIASES
    later = SCerevisiaeGraph.parse_genome(_StubGenome(None))
    # the finding: the attribute exists only because parse_genome set it on the class
    assert later.alias_to_systematic is ALIASES  # type: ignore[attr-defined]


def test_strip_string_prefix_removes_only_a_leading_taxon_prefix(
    sgd_graph: SCerevisiaeGraph,
) -> None:
    """``4932.`` is stripped at the start and nowhere else."""
    assert sgd_graph.strip_string_prefix("4932.YAL001C") == "YAL001C"
    assert sgd_graph.strip_string_prefix("YAL001C") == "YAL001C"
    assert sgd_graph.strip_string_prefix("x4932.Y") == "x4932.Y"


def test_save_and_load_graph_round_trip_by_root_type_and_report_a_missing_file(
    tmp_path: Path, sgd_graph: SCerevisiaeGraph, caplog: pytest.LogCaptureFixture
) -> None:
    """A GeneGraph and a plain nx.Graph pickle under ``<root>/graph/<name>.pkl``."""
    plain = nx.Graph()
    plain.add_edge("YAL001C", "YAL002W", weight=3)
    sgd_graph.save_graph(
        GeneGraph(name="toy", graph=plain, max_gene_set=GENES), "G_toy"
    )
    sgd_graph.save_graph(plain, "G_plain", root_type="string")
    assert (tmp_path / "sgd" / "graph" / "G_toy.pkl").exists()
    assert (tmp_path / "string" / "graph" / "G_plain.pkl").exists()
    toy = sgd_graph.load_graph("G_toy")
    assert isinstance(toy, GeneGraph)
    assert toy.name == "toy"
    assert list(toy.graph.edges(data=True)) == [("YAL001C", "YAL002W", {"weight": 3})]
    back = sgd_graph.load_graph("G_plain", root_type="string")
    assert isinstance(back, nx.Graph)
    assert type(back) is nx.Graph
    assert list(back.edges(data=True)) == [("YAL001C", "YAL002W", {"weight": 3})]
    with caplog.at_level(logging.WARNING, logger="torchcell.graph.graph"):
        assert sgd_graph.load_graph("missing", root_type="tflink") is None
    assert caplog.messages == [
        f"Graph file {tmp_path / 'tflink' / 'graph' / 'missing.pkl'} not found!"
    ]


def test_build_gene_multigraph_loads_only_the_requested_graphs_and_rejects_unknown_names(
    sgd_graph: SCerevisiaeGraph,
) -> None:
    """The multigraph holds the property objects themselves; None in gives None out."""
    multi = build_gene_multigraph(sgd_graph, ["physical", "genetic"])
    assert list(multi) == ["genetic", "physical"]
    assert multi["physical"] is sgd_graph.G_physical
    assert multi["genetic"] is sgd_graph.G_genetic
    assert build_gene_multigraph(sgd_graph, None) is None
    message = "Invalid graph type(s): bogus, nope. Valid types are: " + ", ".join(
        SCEREVISIAE_GENE_GRAPH_VALID_NAMES
    )
    with pytest.raises(ValueError, match=re.escape(message)):
        build_gene_multigraph(sgd_graph, ["physical", "bogus", "nope"])


def test_gene_graph_map_names_the_twenty_two_lazy_graph_properties() -> None:
    """Three SGD graphs, tflink, and six channels for each of three STRING versions."""
    channels = [
        "neighborhood",
        "fusion",
        "cooccurence",
        "coexpression",
        "experimental",
        "database",
    ]
    assert SCEREVISIAE_GENE_GRAPH_VALID_NAMES == [
        "physical",
        "regulatory",
        "genetic",
        "tflink",
        *[f"string{v}_{c}" for v in ("9_1", "11_0", "12_0") for c in channels],
    ]


def test_download_helpers_skip_an_existing_file_and_reject_an_unknown_string_version(
    tmp_path: Path, sgd_graph: SCerevisiaeGraph, caplog: pytest.LogCaptureFixture
) -> None:
    """The version check precedes any I/O; an existing path is left untouched with a log line."""
    with pytest.raises(ValueError, match="Unsupported STRING version: 10.0"):
        sgd_graph.download_string_data(str(tmp_path / "never.gz"), version="10.0")
    assert not (tmp_path / "never.gz").exists()
    existing = tmp_path / "have.gz"
    existing.write_bytes(b"keep")
    with caplog.at_level(logging.INFO, logger="torchcell.graph.graph"):
        sgd_graph.download_string_data(str(existing), version="12.0")
        sgd_graph.download_tflink_data(str(existing))
    assert existing.read_bytes() == b"keep"
    assert caplog.messages == [
        f"STRING v12.0 data already exists at {existing}, skipping download",
        f"TFLink data already exists at {existing}, skipping download",
    ]


# --------------------------------------------------------------------------- SGD-backed
# These build a real SCerevisiaeGenome / SCerevisiaeGraph from the SGD genome data under
# DATA_ROOT, which is not present in CI. They run only with --data and skip when the
# dataset directory is absent.

_GENOME_DIR = os.path.join(DATA_ROOT, "data/sgd/genome") if DATA_ROOT else None
_needs_sgd = pytest.mark.skipif(
    not (_GENOME_DIR and os.path.exists(_GENOME_DIR)),
    reason="requires SGD genome dataset at $DATA_ROOT/data/sgd/genome (absent in CI)",
)


@pytest.fixture
def get_sample_graph() -> nx.DiGraph:
    """Fixture to generate a sample graph for testing."""
    # The skipif above guarantees this only runs with DATA_ROOT set; narrow it to
    # str so the os.path.join calls below type-check under strict mypy.
    assert DATA_ROOT is not None
    genome = SCerevisiaeGenome(
        genome_root=os.path.join(DATA_ROOT, "data/sgd/genome"),
        go_root=os.path.join(DATA_ROOT, "data/go"),
        overwrite=False,  # overwrite=True races a concurrent rebuild of the shared genome
    )
    graph = SCerevisiaeGraph(
        sgd_root=os.path.join(DATA_ROOT, "data/sgd/genome"),
        string_root=os.path.join(DATA_ROOT, "data/string"),
        tflink_root=os.path.join(DATA_ROOT, "data/tflink"),
        genome=genome,
    )
    return graph.G_go


def check_no_IGI_in_graph(G: nx.DiGraph) -> bool:
    """Utility function to check if any node in G has a gene with the 'IGI' display name."""
    for node in tqdm(G.nodes()):
        if "genes" in G.nodes[node]:
            for v in G.nodes[node]["genes"].values():
                if v["go_details"]["experiment"]["display_name"] == "IGI":
                    # Found an IGI
                    return False
    # No IGI found
    return True


@pytest.mark.data
@_needs_sgd
def test_filter_go_IGI(get_sample_graph: nx.DiGraph) -> None:
    """No IGI evidence remains anywhere in the filtered real GO graph."""
    G_filtered = filter_go_IGI(get_sample_graph)
    assert check_no_IGI_in_graph(G_filtered), (
        "Found 'IGI' display name in the filtered graph."
    )


def check_no_genes_after_date(G: nx.DiGraph, cutoff_date: str) -> bool:
    """Utility function to check if any node in G has a gene with a date after the cutoff."""
    for node in tqdm(G.nodes()):
        if "genes" in G.nodes[node]:
            for v in G.nodes[node]["genes"].values():
                gene_date = datetime.strptime(
                    v["go_details"]["date_created"], "%Y-%m-%d"
                )
                cutoff = datetime.strptime(cutoff_date, "%Y-%m-%d")
                if gene_date > cutoff:
                    # Found a gene after the cutoff date
                    return False
    # No genes found after the cutoff date
    return True


@pytest.mark.data
@_needs_sgd
def test_filter_by_date(get_sample_graph: nx.DiGraph) -> None:
    """No annotation dated after the cutoff remains in the filtered real GO graph."""
    cutoff_date = "2018-02-01"
    G_filtered = filter_by_date(get_sample_graph, cutoff_date)
    assert check_no_genes_after_date(G_filtered, cutoff_date), (
        f"Found genes annotated after {cutoff_date} in the filtered graph."
    )
