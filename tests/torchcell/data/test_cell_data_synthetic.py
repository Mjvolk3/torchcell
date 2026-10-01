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

Metabolism, bipartite path (``incidence_graphs["metabolism_bipartite"]``): reactions
``r_B`` (subsystem Growth, genes YAL001C and the unknown YZZ999W) and ``r_A``
(Glycolysis, no genes) over metabolites ``m_x``, ``m_y``, ``m_z``. Sorted, r_A is 0 and
r_B is 1; m_x, m_y, m_z are 0, 1, 2. networkx reports each edge from its first-inserted
endpoint and walks nodes in insertion order, so with r_B inserted before r_A the edges
come out (r_B, m_x), (r_B, m_y), (r_A, m_y), (r_A, m_z): ``hyperedge_index`` [[1, 1, 0,
0], [0, 1, 1, 2]] and signed stoichiometry, reactant negative and product positive,
[-1.0, 2.0, -1.0, 0.5]. ``w_growth`` is [0.0, 1.0]. The GPR edge is gene 0 -> reaction
1; the unknown gene is index -1 in ``reaction_to_genes_indices`` and dropped from the
GPR edge.

Metabolism, hypergraph path (``_process_metabolism_hypergraph``): reactions ``r1``
(m_a -1, m_b +2, gene YAL002W) then ``r0`` (m_b -1, m_c +1, unknown gene) in insertion
order, metabolites sorted m_a, m_b, m_c: ``hyperedge_index`` [[0, 1, 1, 2], [0, 0, 1,
1]], stoichiometry [-1.0, 2.0, -1.0, 1.0], GPR gene 1 -> reaction 0.
"""

import hypernetx as hnx
import networkx as nx
import pytest
import torch
from sortedcontainers import SortedDict

from torchcell.data.cell_data import (
    _process_metabolism_hypergraph,
    compute_strata,
    to_cell_data,
)
from torchcell.data.hetero_data import HeteroData
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
    # add_remaining_self_loops appends the missing loops after the declared edge
    assert edge_index.tolist() == [[0, 0, 1, 2], [1, 0, 1, 2]]
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
    in the reversed graph, so it gets stratum 0. The cycle fallback collapses {x, y} into
    stratum 1 and puts leaf, a child of the cycle, after it at 2 (it used to share the
    cycle's stratum 1; issue #538).
    """
    graph = nx.DiGraph([("leaf", "root"), ("x", "y"), ("y", "x"), ("leaf", "x")])
    assert compute_strata(graph) == {"root": 0, "x": 1, "y": 1, "leaf": 2}


def test_to_cell_data_maps_regulatory_to_its_relation_and_keeps_other_names() -> None:
    """``regulatory`` becomes ``regulatory_interaction``; ``tflink`` stays ``tflink``.
    The directed edge YAL002W -> YAL001C is (1, 0); the undirected YAL003W--YAL001C
    edge is reported from YAL001C, the first-inserted endpoint, as (0, 2). With self
    loops the missing loops follow in node order; without them only the declared edge
    remains and ``num_edges`` is 1.
    """
    multigraph = _multigraph(with_physical=False)
    regulatory = nx.DiGraph()
    regulatory.add_nodes_from(GENES)
    regulatory.add_edge("YAL002W", "YAL001C")
    tflink = nx.Graph()
    tflink.add_nodes_from(GENES)
    tflink.add_edge("YAL003W", "YAL001C")
    multigraph.graphs["regulatory"] = GeneGraph(
        name="regulatory", graph=regulatory, max_gene_set=GENES
    )
    multigraph.graphs["tflink"] = GeneGraph(
        name="tflink", graph=tflink, max_gene_set=GENES
    )

    with_loops = to_cell_data(multigraph)
    assert with_loops.edge_types == [
        ("gene", "regulatory_interaction", "gene"),
        ("gene", "tflink", "gene"),
    ]
    assert with_loops["gene", "regulatory_interaction", "gene"].edge_index.tolist() == [
        [1, 0, 1, 2],
        [0, 0, 1, 2],
    ]
    assert with_loops["gene", "tflink", "gene"].edge_index.tolist() == [
        [0, 0, 1, 2],
        [2, 0, 1, 2],
    ]

    bare = to_cell_data(multigraph, add_remaining_gene_self_loops=False)
    assert bare["gene", "regulatory_interaction", "gene"].edge_index.tolist() == [
        [1],
        [0],
    ]
    assert bare["gene", "tflink", "gene"].edge_index.tolist() == [[0], [2]]
    assert bare["gene", "tflink", "gene"].num_edges == 1


def test_to_cell_data_concatenates_embedding_graphs_and_zero_fills_missing_genes() -> (
    None
):
    """Graphs are visited in SortedDict order (e1 before e2). e1 gives YAL002W a 1-dim
    embedding [7.0] and lists YAL001C without one; e2 gives YAL001C [1.0, 10.0] and
    YAL003W [2.0, 11.0]. x is the column concatenation, zero where a gene has none.
    """
    multigraph = _multigraph(with_physical=False)
    e1 = nx.Graph()
    e1.add_node("YAL002W", embedding=torch.tensor([7.0]))
    e1.add_node("YAL001C")
    e2 = nx.Graph()
    e2.add_node("YAL001C", embedding=torch.tensor([1.0, 10.0]))
    e2.add_node("YAL003W", embedding=torch.tensor([2.0, 11.0]))
    multigraph.graphs["e1"] = GeneGraph(name="e1", graph=e1, max_gene_set=GENES)
    multigraph.graphs["e2"] = GeneGraph(name="e2", graph=e2, max_gene_set=GENES)
    data = to_cell_data(multigraph)
    assert data["gene"].x.tolist() == [
        [0.0, 1.0, 10.0],
        [7.0, 0.0, 0.0],
        [0.0, 2.0, 11.0],
    ]
    assert data["gene"].x.dtype == torch.float


def _bipartite(reactions_first: bool) -> nx.Graph:
    """Two reactions over three metabolites; node insertion order is the parameter."""
    graph = nx.Graph()

    def add_reactions() -> None:
        graph.add_node(
            "r_B",
            node_type="reaction",
            subsystem="Growth",
            genes=["YAL001C", "YZZ999W"],
        )
        graph.add_node("r_A", node_type="reaction", subsystem="Glycolysis", genes=[])

    def add_metabolites() -> None:
        for m in ["m_x", "m_y", "m_z"]:
            graph.add_node(m, node_type="metabolite")

    if reactions_first:
        add_reactions()
        add_metabolites()
    else:
        add_metabolites()
        add_reactions()
    graph.add_edge("r_B", "m_x", edge_type="reactant", stoichiometry=1.0)
    graph.add_edge("r_B", "m_y", edge_type="product", stoichiometry=2.0)
    graph.add_edge("r_A", "m_y", edge_type="reactant", stoichiometry=1.0)
    graph.add_edge("r_A", "m_z", edge_type="product", stoichiometry=0.5)
    return graph


def test_to_cell_data_metabolism_bipartite_signs_stoichiometry_and_links_genes() -> (
    None
):
    """Sorted node ids, signed stoichiometry in networkx edge order, w_growth from the
    subsystem, and a GPR edge only for the gene the base graph knows.
    """
    data = to_cell_data(
        _multigraph(with_physical=False),
        incidence_graphs={"metabolism_bipartite": _bipartite(reactions_first=True)},
    )
    assert data.node_types == ["gene", "metabolite", "reaction"]
    assert data.edge_types == [
        ("reaction", "rmr", "metabolite"),
        ("gene", "gpr", "reaction"),
    ]
    assert data["metabolite"].num_nodes == 3
    assert data["metabolite"].node_ids == ["m_x", "m_y", "m_z"]
    assert data["reaction"].num_nodes == 2
    assert data["reaction"].node_ids == ["r_A", "r_B"]
    assert data["reaction"].w_growth.tolist() == [0.0, 1.0]

    rmr = data["reaction", "rmr", "metabolite"]
    assert rmr.hyperedge_index.tolist() == [[1, 1, 0, 0], [0, 1, 1, 2]]
    assert rmr.stoichiometry.tolist() == [-1.0, 2.0, -1.0, 0.5]
    assert rmr.stoichiometry.dtype == torch.float
    assert rmr.num_edges == 4
    assert rmr.reaction_to_genes == {1: ["YAL001C", "YZZ999W"]}
    assert rmr.reaction_to_genes_indices == {1: [0, -1]}

    gpr = data["gene", "gpr", "reaction"]
    assert gpr.hyperedge_index.tolist() == [[0], [1]]
    assert gpr.num_edges == 1


def test_metabolism_bipartite_drops_every_edge_when_metabolites_are_inserted_first() -> (
    None
):
    """Finding: ``_process_metabolism_bipartite`` keeps an edge only when networkx
    reports it reaction-first (``cell_data.py:514``), and networkx reports an undirected
    edge from whichever endpoint was inserted first. With the metabolite nodes added
    before the reactions no ``rmr`` edge type is created at all, while the node sets,
    ``w_growth`` and the GPR edge are still built. The live graph keeps its edges
    because ``yeast_GEM`` builds an ``nx.DiGraph`` (yeast_GEM.py:257) and stores every
    edge as reaction -> metabolite (406-438), so the stored direction, not the node
    insertion order, is what satisfies the check.
    """
    data = to_cell_data(
        _multigraph(with_physical=False),
        incidence_graphs={"metabolism_bipartite": _bipartite(reactions_first=False)},
    )
    assert data.edge_types == [("gene", "gpr", "reaction")]
    assert data["reaction"].node_ids == ["r_A", "r_B"]
    assert data["reaction"].w_growth.tolist() == [0.0, 1.0]
    assert data["gene", "gpr", "reaction"].hyperedge_index.tolist() == [[0], [1]]


def _hypergraph() -> hnx.Hypergraph:
    """r1 (m_a -1, m_b +2, gene YAL002W) then r0 (m_b -1, m_c +1, an unknown gene)."""
    properties = {
        "r1": {
            "genes": {"YAL002W"},
            "stoich_coefficient-m_a": -1.0,
            "stoich_coefficient-m_b": 2.0,
        },
        "r0": {
            "genes": {"YZZ999W"},
            "stoich_coefficient-m_b": -1.0,
            "stoich_coefficient-m_c": 1.0,
        },
    }
    return hnx.Hypergraph(
        {"r1": ["m_a", "m_b"], "r0": ["m_b", "m_c"]}, edge_properties=properties
    )


def test_to_cell_data_never_dispatches_the_metabolism_hypergraph() -> None:
    """Finding: ``to_cell_data`` reads only the ``metabolism_bipartite`` and
    ``gene_ontology`` keys (``cell_data.py:98-105``); a hypergraph passed under
    ``metabolism_hypergraph`` is silently ignored and ``_process_metabolism_hypergraph``
    has no caller in the module.
    """
    data = to_cell_data(
        _multigraph(with_physical=False),
        incidence_graphs={"metabolism_hypergraph": _hypergraph()},
    )
    assert data.node_types == ["gene"]
    assert data.edge_types == []


def test_process_metabolism_hypergraph_indexes_metabolites_reactions_and_genes() -> (
    None
):
    """Finding: ``num_edges`` on the metabolite hyperedge is the reaction count plus one
    (``cell_data.py:171`` adds 1 to the number of distinct reaction indices), so two
    reactions report 3. Everything else follows the docstring: metabolites sorted, reactions
    in insertion order, stoichiometry read per metabolite, the unknown gene kept as -1 in
    ``reaction_to_genes_indices`` and dropped from the GPR edge.
    """
    data = HeteroData()
    _process_metabolism_hypergraph(
        data, _hypergraph(), {"YAL001C": 0, "YAL002W": 1, "YAL003W": 2}
    )
    assert data["metabolite"].num_nodes == 3
    assert data["metabolite"].node_ids == ["m_a", "m_b", "m_c"]
    assert data["reaction"].num_nodes == 2
    assert data["reaction"].node_ids == [0, 1]

    hyper = data["metabolite", "reaction", "metabolite"]
    assert hyper.hyperedge_index.tolist() == [[0, 1, 1, 2], [0, 0, 1, 1]]
    assert hyper.stoichiometry.tolist() == [-1.0, 2.0, -1.0, 1.0]
    assert hyper.num_edges == 3
    assert hyper.reaction_to_genes == {0: ["YAL002W"], 1: ["YZZ999W"]}
    assert hyper.reaction_to_genes_indices == {0: [1], 1: [-1]}

    gpr = data["gene", "gpr", "reaction"]
    assert gpr.hyperedge_index.tolist() == [[1], [0]]
    assert gpr.num_edges == 1
