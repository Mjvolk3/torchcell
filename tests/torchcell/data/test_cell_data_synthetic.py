# tests/torchcell/data/test_cell_data_synthetic.py
# [[tests.torchcell.data.test_cell_data_synthetic]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/data/test_cell_data_synthetic.py
"""``to_cell_data`` and ``compute_strata`` on hand-built graphs, no data root.

``tests/torchcell/data/test_cell_data.py`` exercises the same module on the real
sample batch behind ``--data``; this file pins the pure parts. Three genes sorted
YAL001C < YAL002W < YAL003W give node indices 0, 1, 2. A physical edge YAL001C--YAL002W
plus ``add_remaining_self_loops`` gives the four edges (0,1), (0,0), (1,1), (2,2) under the
relation name ``physical_interaction``. GO edges point child -> parent, so for c -> a ->
root, b -> root the strata are root: 0, a: 1, b: 1, c: 2. DCell iterates the strata in
descending order, so the leaves are processed first and the root last.
"""

import networkx as nx
import pytest
import torch
from sortedcontainers import SortedDict

from torchcell.data.cell_data import compute_strata, to_cell_data
from torchcell.graph.graph import GeneGraph, GeneMultiGraph
from torchcell.sequence import GeneSet

UNSORTED = ["YAL003W", "YAL001C", "YAL002W"]  # a GeneSet would already be sorted
GENES = GeneSet(UNSORTED)


def _multigraph(with_physical: bool = True) -> GeneMultiGraph:
    base = nx.Graph()
    base.add_nodes_from(UNSORTED)
    graphs = {"base": GeneGraph(name="base", graph=base, max_gene_set=GENES)}
    if with_physical:
        physical = nx.Graph()
        physical.add_nodes_from(GENES)
        physical.add_edge("YAL001C", "YAL002W")
        graphs["physical"] = GeneGraph(
            name="physical", graph=physical, max_gene_set=GENES
        )
    return GeneMultiGraph(graphs=SortedDict(graphs))


def test_to_cell_data_sorts_genes_and_adds_remaining_self_loops() -> None:
    """node_ids are sorted from the base graph's insertion order (YAL003W first there);
    physical edges get every missing self loop; x is [3, 0].
    """
    multigraph = _multigraph()
    assert list(multigraph.graphs["base"].graph.nodes()) == UNSORTED
    data = to_cell_data(multigraph)
    assert data["gene"].num_nodes == 3
    assert data["gene"].node_ids == ["YAL001C", "YAL002W", "YAL003W"]
    assert data["gene"].x.shape == (3, 0)
    edge_index = data["gene", "physical_interaction", "gene"].edge_index
    edges = sorted(map(tuple, edge_index.t().tolist()))
    assert edges == [(0, 0), (0, 1), (1, 1), (2, 2)]
    assert data["gene", "physical_interaction", "gene"].num_edges == 4


def test_to_cell_data_without_self_loops_keeps_only_the_declared_edge() -> None:
    """add_remaining_gene_self_loops=False leaves the single (0, 1) edge."""
    data = to_cell_data(_multigraph(), add_remaining_gene_self_loops=False)
    edge_index = data["gene", "physical_interaction", "gene"].edge_index
    assert edge_index.tolist() == [[0], [1]]


def test_to_cell_data_uses_node_embeddings_from_an_edgeless_graph() -> None:
    """An edgeless graph carrying ``embedding`` node attributes fills x column-wise."""
    multigraph = _multigraph(with_physical=False)
    embedding = nx.Graph()
    for i, gene in enumerate(["YAL001C", "YAL002W", "YAL003W"]):
        embedding.add_node(gene, embedding=torch.tensor([float(i), 10.0 * i]))
    multigraph.graphs["fudt"] = GeneGraph(
        name="fudt", graph=embedding, max_gene_set=GENES
    )
    data = to_cell_data(multigraph)
    torch.testing.assert_close(
        data["gene"].x, torch.tensor([[0.0, 0.0], [1.0, 10.0], [2.0, 20.0]])
    )


def test_to_cell_data_requires_a_base_graph() -> None:
    """The gene index is defined by the base graph; without it the call is refused."""
    multigraph = _multigraph()
    del multigraph.graphs["base"]
    with pytest.raises(ValueError, match="must contain a 'base' graph"):
        to_cell_data(multigraph)


def test_compute_strata_numbers_the_root_zero_and_children_by_depth() -> None:
    """GO edges point child -> parent (create_G_go): c -> a -> root, b -> root gives
    {root: 0, a: 1, b: 1, c: 2}. Stratum 0 is the root; DCell walks the strata in
    descending order so the deepest terms are computed first. The function's own
    docstring says leaves are stratum 0, which is not what it does.
    """
    dag = nx.DiGraph([("a", "root"), ("b", "root"), ("c", "a")])
    assert compute_strata(dag) == {"root": 0, "a": 1, "b": 1, "c": 2}


def test_compute_strata_places_a_cyclic_component_in_one_stratum_after_the_dag() -> (
    None
):
    """Leaf -> root, leaf -> x, x <-> y: root is the only node that ever reaches in-degree 0
    in the reversed graph, so it gets stratum 0 and the cycle fallback assigns the whole
    remaining component (leaf, x, y; no sinks among them) to stratum 1.
    """
    graph = nx.DiGraph([("leaf", "root"), ("x", "y"), ("y", "x"), ("leaf", "x")])
    assert compute_strata(graph) == {"root": 0, "leaf": 1, "x": 1, "y": 1}
