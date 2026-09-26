# tests/torchcell/graph/test_gene_graph.py
# [[tests.torchcell.graph.test_gene_graph]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/graph/test_gene_graph.py
"""``GeneGraph`` / ``GeneMultiGraph`` and the four GO filters on hand-built graphs.

No data root: the graphs are three or four nodes with the ``gene_set`` / ``genes``
attributes the filters read. Each filter's expected node set is worked in its test.
``tests/torchcell/graph/test_graph.py`` keeps the SGD-backed tests behind ``--data``.
"""

import logging
from typing import Any

import networkx as nx
import pytest
from pydantic import ValidationError
from sortedcontainers import SortedDict

from torchcell.graph.graph import (
    GeneGraph,
    GeneMultiGraph,
    filter_by_contained_genes,
    filter_by_date,
    filter_go_IGI,
    filter_redundant_terms,
)
from torchcell.sequence import GeneSet

GENES = GeneSet(["YAL001C", "YAL002W", "YAL003W"])


def _gene_graph(name: str, edges: list[tuple[str, str]]) -> GeneGraph:
    graph = nx.Graph()
    graph.add_nodes_from(GENES)
    graph.add_edges_from(edges)
    return GeneGraph(name=name, graph=graph, max_gene_set=GENES)


def test_gene_graph_forwards_networkx_calls_and_reprs_its_counts() -> None:
    """Attribute access reaches the wrapped graph; the repr states nodes and edges."""
    gene_graph = _gene_graph("physical", [("YAL001C", "YAL002W")])
    assert gene_graph.number_of_nodes() == 3
    assert gene_graph.number_of_edges() == 1
    assert gene_graph.has_edge("YAL001C", "YAL002W")
    assert repr(gene_graph) == "GeneGraph(name='physical', nodes=3, edges=1)"


def test_gene_graph_accepts_nodes_outside_the_gene_set_without_a_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Finding: the ``graph`` validator runs before ``max_gene_set`` is parsed (field order),
    so its "nodes not in max_gene_set" warning never fires. Two foreign nodes are kept
    silently; this pins that until the validator is moved to a model validator.
    """
    graph = nx.Graph()
    graph.add_nodes_from(["YAL001C", "YZZ999W", "YZZ998W"])
    with caplog.at_level(logging.WARNING, logger="torchcell.graph.graph"):
        gene_graph = GeneGraph(name="g", graph=graph, max_gene_set=GENES)
    assert gene_graph.number_of_nodes() == 3
    assert caplog.text == ""


def test_gene_graph_requires_a_networkx_graph() -> None:
    """The strict model rejects a non-graph for the graph field."""
    with pytest.raises(ValidationError):
        GeneGraph(name="g", graph="not a graph", max_gene_set=GENES)


def test_multigraph_is_a_sorted_mapping_of_gene_graphs() -> None:
    """Names iterate sorted, lookups and membership work, repr lists every graph."""
    physical = _gene_graph("physical", [("YAL001C", "YAL002W")])
    base = _gene_graph("base", [])
    multi = GeneMultiGraph(graphs=SortedDict({"physical": physical, "base": base}))
    assert list(multi) == ["base", "physical"]
    assert list(multi.keys()) == ["base", "physical"]
    assert multi["physical"] is physical
    assert "base" in multi and "genetic" not in multi
    assert len(multi) == 2
    assert [name for name, _ in multi.items()] == ["base", "physical"]
    assert list(multi.values()) == [base, physical]
    assert repr(multi) == (
        "GeneMultiGraph(\n  base: 3 nodes, 0 edges\n  physical: 3 nodes, 1 edges\n)"
    )


def _go_graph() -> nx.DiGraph:
    """GO edges point child -> parent as ``create_G_go`` builds them: c -> a -> root, b -> root.

    Gene sets: c {g1}, a {g1}, b {g2, g3}, root {g1, g2, g3}.
    """

    def detail(date: str, experiment: str = "IDA") -> dict[str, Any]:
        return {
            "go_details": {
                "date_created": date,
                "experiment": {"display_name": experiment},
            }
        }

    graph = nx.DiGraph()
    graph.add_edges_from([("a", "root"), ("b", "root"), ("c", "a")])
    graph.nodes["c"].update(gene_set={"g1"}, genes={"g1": detail("2015-01-01")})
    graph.nodes["a"].update(gene_set={"g1"}, genes={"g1": detail("2015-01-01")})
    graph.nodes["b"].update(
        gene_set={"g2", "g3"},
        genes={"g2": detail("2021-06-01", "IGI"), "g3": detail("2010-01-01")},
    )
    graph.nodes["root"].update(
        gene_set={"g1", "g2", "g3"},
        genes={
            "g1": detail("2015-01-01"),
            "g2": detail("2021-06-01", "IGI"),
            "g3": detail("2010-01-01"),
        },
    )
    return graph


def test_filter_by_date_drops_late_annotations_and_emptied_terms() -> None:
    """Cutoff 2016: g2 (2021) leaves every term; nothing empties. Cutoff 2012 empties a and c."""
    original = _go_graph()
    filtered = filter_by_date(original, "2016-01-01")
    assert set(filtered.nodes) == {"root", "a", "b", "c"}
    assert filtered.nodes["b"]["gene_set"] == {"g3"}
    assert set(filtered.nodes["root"]["genes"]) == {"g1", "g3"}
    assert original.nodes["b"]["gene_set"] == {"g2", "g3"}  # input untouched
    early = filter_by_date(original, "2012-01-01")
    assert set(early.nodes) == {"root", "b"}
    assert early.nodes["b"]["gene_set"] == {"g3"}
    assert set(early.edges) == {("b", "root")}


def test_filter_go_igi_removes_only_genetic_interaction_evidence() -> None:
    """g2 is IGI-annotated in b and root; both lose it, nothing is emptied."""
    filtered = filter_go_IGI(_go_graph())
    assert set(filtered.nodes) == {"root", "a", "b", "c"}
    assert filtered.nodes["b"]["gene_set"] == {"g3"}
    assert filtered.nodes["root"]["gene_set"] == {"g1", "g3"}
    assert filtered.nodes["a"]["gene_set"] == {"g1"}


def test_filter_redundant_terms_removes_the_broader_term_with_an_identical_gene_set() -> (
    None
):
    """A and c both hold {g1}: a (the parent) is removed and c is rewired to root.

    The docstring says the term equal to "one of its parents" goes, but the walk starts
    at the roots (nodes with no successors under child -> parent edges) and removes the
    node whose PREDECESSOR (child) matches, so the broader term is the one dropped.
    """
    filtered = filter_redundant_terms(_go_graph())
    assert set(filtered.nodes) == {"root", "b", "c"}
    assert set(filtered.edges) == {("b", "root"), ("c", "root")}


def test_filter_by_contained_genes_rewires_around_small_terms() -> None:
    """Containment is the union over descendants: n=2 drops a and c, keeps root and b."""
    filtered = filter_by_contained_genes(_go_graph(), n=2, gene_set={"g1", "g2", "g3"})
    assert set(filtered.nodes) == {"root", "b"}
    assert set(filtered.edges) == {("b", "root")}
    restricted = filter_by_contained_genes(_go_graph(), n=1, gene_set={"g1"})
    assert set(restricted.nodes) == {"root", "a", "c"}
    assert set(restricted.edges) == {("c", "a"), ("a", "root")}
