# tests/torchcell/data/test_graph_processor_subgraph.py
# [[tests.torchcell.data.test_graph_processor_subgraph]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/data/test_graph_processor_subgraph.py
"""The subgraph processors on a five-gene path graph with a metabolism block, exact tensors.

Fixture. Genes YAL001C..YAL005C sort to indices 0..4. The physical graph is the path
0-1-2-3-4; ``to_cell_data`` writes each undirected edge once in insertion order and
appends the missing self loops, so the physical ``edge_index`` is

    src [0, 1, 2, 3, 0, 1, 2, 3, 4]
    dst [1, 2, 3, 4, 0, 1, 2, 3, 4]      (positions 0..8)

The bipartite block has reactions r1 (genes YAL001C, YAL002W; subsystem Growth), r2
(gene YAL004W) and r3 (no genes) at indices 0, 1, 2 and metabolites m_a, m_b, m_c at
0, 1, 2. Edges in insertion order, reactants negative:

    rmr hyperedge_index  reaction [0, 0, 1, 1, 2, 2]  metabolite [0, 1, 1, 2, 2, 0]
    stoichiometry        [-1, 2, -1, 1, -1, 1]
    gpr hyperedge_index  gene [0, 1, 3]  reaction [0, 0, 1]
    w_growth             [1, 0, 0]

Perturbing YAL001C (gene 0) keeps genes [1, 2, 3, 4], so the gene map is
{1: 0, 2: 1, 3: 2, 4: 3}; every edge touching 0 (positions 0 and 4) is dropped and the
seven survivors relabel to src [0, 1, 2, 0, 1, 2, 3] dst [1, 2, 3, 0, 1, 2, 3]. Reaction
r1 loses one of its genes and is removed; r2 (gene kept) and r3 (no genes) survive, so
``valid_reactions`` is [1, 2] with reaction map {1: 0, 2: 1}. Perturbing YAL001C and
YAL004W across two records removes r1 and r2 and leaves only the gene-free r3.

``LazySubgraphRepresentation`` keeps every node and edge and emits masks instead; the
mapping between its output and ``SubgraphRepresentation`` is asserted exactly in
``test_lazy_masks_map_onto_the_subgraph_output``. ``NeighborSubgraphRepresentation.process``
raises on this input (see the Finding); its k-hop pieces are exercised through the
gene-info shape they accept.
"""

from typing import Any

import networkx as nx
import pytest
import torch
from sortedcontainers import SortedDict

from torchcell.data.cell_data import to_cell_data
from torchcell.data.graph_processor import (
    IncidenceSubgraphRepresentation,
    LazySubgraphRepresentation,
    NeighborSubgraphRepresentation,
    SubgraphRepresentation,
)
from torchcell.data.hetero_data import HeteroData
from torchcell.datamodels.schema import (
    Environment,
    FitnessExperiment,
    FitnessExperimentReference,
    FitnessPhenotype,
    GeneInteractionPhenotype,
    Genotype,
    KanMxDeletionPerturbation,
    Media,
    ReferenceGenome,
)
from torchcell.graph.graph import GeneGraph, GeneMultiGraph
from torchcell.sequence import GeneSet

GENE_NAMES = ["YAL001C", "YAL002W", "YAL003W", "YAL004W", "YAL005C"]
GENES = GeneSet(GENE_NAMES)
PHENOTYPES: list[Any] = [FitnessPhenotype]
ENVIRONMENT = Environment(media=Media(name="YPD", state="solid", is_synthetic=False))
PHYSICAL = ("gene", "physical_interaction", "gene")
GPR = ("gene", "gpr", "reaction")
RMR = ("reaction", "rmr", "metabolite")
FULL_SRC = [0, 1, 2, 3, 0, 1, 2, 3, 4]
FULL_DST = [1, 2, 3, 4, 0, 1, 2, 3, 4]


def _bipartite() -> nx.Graph:
    """r1 {YAL001C, YAL002W} Growth, r2 {YAL004W}, r3 gene-free; three metabolites."""
    graph = nx.Graph()
    graph.add_node(
        "r1", node_type="reaction", genes=["YAL001C", "YAL002W"], subsystem="Growth"
    )
    graph.add_node("r2", node_type="reaction", genes=["YAL004W"], subsystem="Other")
    graph.add_node("r3", node_type="reaction", genes=[], subsystem="Other")
    for metabolite in ["m_a", "m_b", "m_c"]:
        graph.add_node(metabolite, node_type="metabolite")
    graph.add_edge("r1", "m_a", edge_type="reactant", stoichiometry=1.0)
    graph.add_edge("r1", "m_b", edge_type="product", stoichiometry=2.0)
    graph.add_edge("r2", "m_b", edge_type="reactant", stoichiometry=1.0)
    graph.add_edge("r2", "m_c", edge_type="product", stoichiometry=1.0)
    graph.add_edge("r3", "m_c", edge_type="reactant", stoichiometry=1.0)
    graph.add_edge("r3", "m_a", edge_type="product", stoichiometry=1.0)
    return graph


def _cell_graph() -> HeteroData:
    """Path 0-1-2-3-4 plus self loops, with the bipartite metabolism block."""
    base = nx.Graph()
    base.add_nodes_from(GENES)
    physical = nx.Graph()
    physical.add_nodes_from(GENES)
    for src, dst in zip(GENE_NAMES[:-1], GENE_NAMES[1:], strict=True):
        physical.add_edge(src, dst)
    multigraph = GeneMultiGraph(
        graphs=SortedDict(
            {
                "base": GeneGraph(name="base", graph=base, max_gene_set=GENES),
                "physical": GeneGraph(
                    name="physical", graph=physical, max_gene_set=GENES
                ),
            }
        )
    )
    cell_graph = to_cell_data(
        multigraph, incidence_graphs={"metabolism_bipartite": _bipartite()}
    )
    assert cell_graph[PHYSICAL].edge_index.tolist() == [FULL_SRC, FULL_DST]
    assert cell_graph[GPR].hyperedge_index.tolist() == [[0, 1, 3], [0, 0, 1]]
    assert cell_graph[RMR].hyperedge_index.tolist() == [
        [0, 0, 1, 1, 2, 2],
        [0, 1, 1, 2, 2, 0],
    ]
    assert cell_graph["reaction"].w_growth.tolist() == [1.0, 0.0, 0.0]
    return cell_graph


def _record(
    genes: list[str], fitness: float, se: float | None = None
) -> dict[str, Any]:
    perturbations = [
        KanMxDeletionPerturbation(systematic_gene_name=g, perturbed_gene_name=g)
        for g in genes
    ]
    genotype = Genotype(perturbations=perturbations)  # type: ignore[arg-type]
    return {
        "experiment": FitnessExperiment(
            dataset_name="toy",
            genotype=genotype,
            environment=ENVIRONMENT,
            phenotype=FitnessPhenotype(fitness=fitness, fitness_se=se),
        ),
        "experiment_reference": FitnessExperimentReference(
            dataset_name="toy",
            genome_reference=ReferenceGenome(
                species="Saccharomyces cerevisiae", strain="S288C"
            ),
            environment_reference=ENVIRONMENT,
            phenotype_reference=FitnessPhenotype(fitness=1.0),
        ),
    }


def _store(data: HeteroData, key: Any) -> dict[str, Any]:
    """A node or edge store as plain Python, with ids_pert sorted (it comes from a set)."""
    items = {k: (v.tolist() if torch.is_tensor(v) else v) for k, v in data[key].items()}
    if "ids_pert" in items:
        items["ids_pert"] = sorted(items["ids_pert"])
    return items


SINGLE = [_record(["YAL001C"], 0.9)]
DOUBLE = [_record(["YAL001C"], 0.9, se=0.05), _record(["YAL004W"], 0.4)]


def test_subgraph_removes_the_perturbed_gene_and_relabels_the_survivors() -> None:
    """Perturbing gene 0 keeps [1, 2, 3, 4]; edges at positions 0 and 4 go; the rest
    relabel through {1: 0, 2: 1, 3: 2, 4: 3}. x has zero columns, so x is [4, 0] and
    x_pert is [1, 0].
    """
    out = SubgraphRepresentation().process(_cell_graph(), PHENOTYPES, SINGLE)
    gene = out["gene"]
    assert gene.node_ids == ["YAL002W", "YAL003W", "YAL004W", "YAL005C"]
    assert gene.num_nodes == 4
    assert gene.ids_pert == ["YAL001C"]
    assert gene.perturbation_indices.tolist() == [0]
    assert gene.pert_mask.tolist() == [True, False, False, False, False]
    assert gene.x.tolist() == [[], [], [], []]
    assert gene.x_pert.tolist() == [[]]
    edges = out[PHYSICAL]
    assert edges.edge_index.tolist() == [[0, 1, 2, 0, 1, 2, 3], [1, 2, 3, 0, 1, 2, 3]]
    assert edges.num_edges == 7
    assert edges.pert_mask.tolist() == [
        True,
        False,
        False,
        False,
        True,
        False,
        False,
        False,
        False,
    ]
    torch.testing.assert_close(gene.phenotype_values, torch.tensor([0.9]))
    assert gene.phenotype_type_indices.tolist() == [0]
    assert gene.phenotype_sample_indices.tolist() == [0]
    assert gene.phenotype_types == ["fitness"]


def test_subgraph_drops_reactions_that_lose_a_gene_and_keeps_gene_free_ones() -> None:
    """Gene 0 sits on r1, so r1 is removed; r2 keeps its gene and r3 has none, so
    valid_reactions is [1, 2] (map {1: 0, 2: 1}). The only surviving gpr edge is
    (gene 3, r2) -> (2, 0); gpr pert_mask marks positions 0 and 1. rmr keeps positions
    2..5 with reactions relabeled to [0, 0, 1, 1]; w_growth [0, 0] is w_growth[[1, 2]].
    Metabolites are never removed.
    """
    out = SubgraphRepresentation().process(_cell_graph(), PHENOTYPES, SINGLE)
    reaction = out["reaction"]
    assert reaction.node_ids == [1, 2]
    assert reaction.num_nodes == 2
    assert reaction.w_growth.tolist() == [0.0, 0.0]
    assert reaction.pert_mask.tolist() == [True, False, False]
    assert out[GPR].hyperedge_index.tolist() == [[2], [0]]
    assert out[GPR].num_edges == 1
    assert out[GPR].pert_mask.tolist() == [True, True, False]
    assert out[RMR].hyperedge_index.tolist() == [[0, 0, 1, 1], [1, 2, 2, 0]]
    assert out[RMR].stoichiometry.tolist() == [-1.0, 1.0, -1.0, 1.0]
    assert out[RMR].num_edges == 4
    assert out["metabolite"].node_ids == ["m_a", "m_b", "m_c"]
    assert out["metabolite"].num_nodes == 3
    assert out["metabolite"].pert_mask.tolist() == [False, False, False]


def test_subgraph_unions_two_records_and_stores_the_statistic_coo() -> None:
    """{YAL001C} and {YAL004W} keep [1, 2, 4] (map {1: 0, 2: 1, 4: 2}): surviving edges
    (1,2), (1,1), (2,2), (4,4) at positions 1, 5, 6, 8 relabel to src [0, 0, 1, 2]
    dst [1, 0, 1, 2]. r1 and r2 each lose a gene, so only r3 survives: no gpr edge, rmr
    positions 4 and 5 relabeled to reaction 0. fitness_se 0.05 on record 0 only.
    """
    out = SubgraphRepresentation().process(_cell_graph(), PHENOTYPES, DOUBLE)
    gene = out["gene"]
    assert gene.node_ids == ["YAL002W", "YAL003W", "YAL005C"]
    assert sorted(gene.ids_pert) == ["YAL001C", "YAL004W"]
    assert gene.perturbation_indices.tolist() == [0, 3]
    assert gene.pert_mask.tolist() == [True, False, False, True, False]
    assert gene.x_pert.tolist() == [[], []]
    assert out[PHYSICAL].edge_index.tolist() == [[0, 0, 1, 2], [1, 0, 1, 2]]
    assert out[PHYSICAL].pert_mask.tolist() == [
        True,
        False,
        True,
        True,
        True,
        False,
        False,
        True,
        False,
    ]
    assert out["reaction"].node_ids == [2]
    assert out["reaction"].pert_mask.tolist() == [True, True, False]
    assert out[GPR].hyperedge_index.tolist() == [[], []]
    assert out[GPR].num_edges == 0
    assert out[GPR].pert_mask.tolist() == [True, True, True]
    assert out[RMR].hyperedge_index.tolist() == [[0, 0], [2, 0]]
    assert out[RMR].stoichiometry.tolist() == [-1.0, 1.0]
    torch.testing.assert_close(gene.phenotype_values, torch.tensor([0.9, 0.4]))
    assert gene.phenotype_sample_indices.tolist() == [0, 1]
    torch.testing.assert_close(gene.phenotype_stat_values, torch.tensor([0.05]))
    assert gene.phenotype_stat_type_indices.tolist() == [0]
    assert gene.phenotype_stat_sample_indices.tolist() == [0]
    assert gene.phenotype_stat_types == ["fitness_se"]


def test_subgraph_accepts_an_empty_batch_and_returns_the_unperturbed_graph() -> None:
    """Finding: unlike Perturbation, Unperturbed and DCellGraphProcessor, which raise
    "Data list is empty", SubgraphRepresentation.process (graph_processor.py:101) takes
    an empty list and returns the whole graph: all 5 genes, all 9 edges, all 3
    reactions, no perturbation, and the NaN phenotype placeholder.
    """
    out = SubgraphRepresentation().process(_cell_graph(), PHENOTYPES, [])
    gene = out["gene"]
    assert gene.node_ids == GENE_NAMES
    assert gene.ids_pert == []
    assert gene.perturbation_indices.tolist() == []
    assert gene.pert_mask.tolist() == [False] * 5
    assert out[PHYSICAL].edge_index.tolist() == [FULL_SRC, FULL_DST]
    assert out[PHYSICAL].pert_mask.tolist() == [False] * 9
    assert out["reaction"].node_ids == [0, 1, 2]
    assert out["reaction"].w_growth.tolist() == [1.0, 0.0, 0.0]
    assert gene.phenotype_values.isnan().tolist() == [True]
    assert gene.phenotype_type_indices.tolist() == [0]
    assert gene.phenotype_sample_indices.tolist() == [0]


def test_subgraph_omits_the_statistic_keys_when_no_record_carries_one() -> None:
    """Finding: with phenotype_info [GeneInteractionPhenotype] and a fitness record, no
    value matches, so phenotype_values is the [nan] placeholder and the four
    phenotype_stat_* keys are absent (graph_processor.py:554 writes them only when a
    statistic was found). Perturbation writes empty tensors in the same case, so a batch
    mixing the two processors' outputs would not collate.
    """
    out = SubgraphRepresentation().process(
        _cell_graph(), [GeneInteractionPhenotype], SINGLE
    )
    gene = out["gene"]
    assert gene.phenotype_values.isnan().tolist() == [True]
    assert gene.phenotype_types == ["gene_interaction"]
    assert "phenotype_stat_values" not in gene
    assert "phenotype_stat_types" not in gene
    assert gene.node_ids == ["YAL002W", "YAL003W", "YAL004W", "YAL005C"]


@pytest.mark.parametrize("data", [SINGLE, DOUBLE], ids=["gene0", "genes0and3"])
def test_incidence_output_equals_the_subgraph_output_store_by_store(
    data: list[dict[str, Any]],
) -> None:
    """The incidence cache is an implementation detail: every node and edge store of
    IncidenceSubgraphRepresentation equals SubgraphRepresentation's on the same input,
    and the gene edge_index is the exact relabeled tensor derived in the module docstring.
    """
    cell_graph = _cell_graph()
    reference = SubgraphRepresentation().process(cell_graph, PHENOTYPES, data)
    incidence = IncidenceSubgraphRepresentation().process(cell_graph, PHENOTYPES, data)
    assert incidence.node_types == ["gene", "reaction", "metabolite"]
    assert incidence.edge_types == [PHYSICAL, GPR, RMR]
    for key in incidence.node_types + incidence.edge_types:
        assert _store(incidence, key) == _store(reference, key), key
    expected_edges = (
        [[0, 1, 2, 0, 1, 2, 3], [1, 2, 3, 0, 1, 2, 3]]
        if data is SINGLE
        else [[0, 0, 1, 2], [1, 0, 1, 2]]
    )
    assert incidence[PHYSICAL].edge_index.tolist() == expected_edges


def test_incidence_cache_lists_edge_positions_per_gene_once_per_self_loop() -> None:
    """Gene 0 touches positions 0 and 4; gene 1 positions 0, 1, 5; ...; gene 4 positions
    3 and 8 (a self loop is recorded once). The property raises before the cache exists.
    """
    processor = IncidenceSubgraphRepresentation()
    with pytest.raises(RuntimeError, match="Incidence cache not built"):
        processor.edge_incidence_cache  # noqa: B018
    processor.build_cache(_cell_graph())
    cache = processor.edge_incidence_cache
    assert list(cache) == [PHYSICAL]
    assert [t.tolist() for t in cache[PHYSICAL]] == [
        [0, 4],
        [0, 1, 5],
        [1, 2, 6],
        [2, 3, 7],
        [3, 8],
    ]


@pytest.mark.parametrize(
    "processor",
    [LazySubgraphRepresentation(), IncidenceSubgraphRepresentation()],
    ids=["lazy", "incidence"],
)
def test_build_cache_halves_the_incidence_count_and_undercounts_self_loops(
    processor: LazySubgraphRepresentation | IncidenceSubgraphRepresentation,
) -> None:
    """Finding: build_cache reports total_edges as (sum of incidence list lengths) // 2
    (graph_processor.py:1322-1329, and the same code at 638-645 in the incidence
    class), assuming every edge is listed twice. The four path edges are listed twice
    (8) and the five self loops once (5), so 13 // 2 = 6 for a graph with 9 edges. A
    second call returns the zero dict without rebuilding.
    """
    cell_graph = _cell_graph()
    first = processor.build_cache(cell_graph)
    assert first["num_edge_types"] == 1
    assert first["total_edges"] == 6
    assert cell_graph[PHYSICAL].num_edges == 9
    assert processor.build_cache(cell_graph) == {
        "total_time_ms": 0.0,
        "num_edge_types": 0,
        "total_edges": 0,
    }


def test_lazy_keeps_every_node_and_edge_by_reference_and_emits_masks() -> None:
    """Finding: the class docstring (graph_processor.py:1239-1246) promises filtered
    node_ids, num_nodes and an x_pert; the code keeps all 5 genes, the same x object,
    and writes no x_pert. Edges: edge_index is the cell graph's tensor (identity), and
    mask is False at positions 0 and 4 (edges touching gene 0).
    """
    cell_graph = _cell_graph()
    out = LazySubgraphRepresentation().process(cell_graph, PHENOTYPES, SINGLE)
    gene = out["gene"]
    assert gene.node_ids == GENE_NAMES
    assert gene.num_nodes == 5
    assert gene.x is cell_graph["gene"].x
    assert "x_pert" not in gene
    assert gene.ids_pert == ["YAL001C"]
    assert gene.perturbation_indices.tolist() == [0]
    assert gene.pert_mask.tolist() == [True, False, False, False, False]
    assert gene.mask.tolist() == [False, True, True, True, True]
    edges = out[PHYSICAL]
    assert edges.edge_index is cell_graph[PHYSICAL].edge_index
    assert edges.num_edges == 9
    assert edges.mask.tolist() == [
        False,
        True,
        True,
        True,
        False,
        True,
        True,
        True,
        True,
    ]


def test_lazy_marks_invalid_reactions_but_keeps_all_of_them() -> None:
    """Perturbing gene 0 invalidates r1 only: reaction pert_mask [T, F, F], mask
    [F, T, T], node_ids [0, 1, 2]. The gpr mask is per gene (False only where the gene
    is deleted: position 0), the rmr mask follows the reaction: positions 0 and 1 (r1)
    are False. w_growth is the cell graph's stored [1, 0, 0], returned whole (issue
    #527; before the fix Lazy recomputed it from a ``subsystem`` attribute the
    ``to_cell_data`` graph does not carry and returned [0, 0, 0]).
    """
    cell_graph = _cell_graph()
    out = LazySubgraphRepresentation().process(cell_graph, PHENOTYPES, SINGLE)
    reaction = out["reaction"]
    assert reaction.node_ids == [0, 1, 2]
    assert reaction.num_nodes == 3
    assert reaction.pert_mask.tolist() == [True, False, False]
    assert reaction.mask.tolist() == [False, True, True]
    assert reaction.w_growth.tolist() == [1.0, 0.0, 0.0]
    assert reaction.w_growth is cell_graph["reaction"].w_growth
    assert out[GPR].hyperedge_index is cell_graph[GPR].hyperedge_index
    assert out[GPR].mask.tolist() == [False, True, True]
    assert out[GPR].num_edges == 3
    assert out[RMR].hyperedge_index is cell_graph[RMR].hyperedge_index
    assert out[RMR].stoichiometry.tolist() == [-1.0, 2.0, -1.0, 1.0, -1.0, 1.0]
    assert out[RMR].mask.tolist() == [False, False, True, True, True, True]
    assert out[RMR].num_edges == 6
    assert out["metabolite"].pert_mask.tolist() == [False, False, False]
    assert out["metabolite"].mask.tolist() == [True, True, True]


def test_lazy_ignores_a_subsystem_list_and_returns_the_stored_w_growth() -> None:
    """A ``subsystem`` list that contradicts the stored tensor (Growth at r3, not r1)
    is not read: the output is the stored w_growth [1, 0, 0] (issue #527, one source).
    """
    cell_graph = _cell_graph()
    cell_graph["reaction"].subsystem = ["Other", "Other", "Growth"]
    out = LazySubgraphRepresentation().process(cell_graph, PHENOTYPES, SINGLE)
    assert out["reaction"].w_growth.tolist() == [1.0, 0.0, 0.0]


def test_lazy_masks_map_onto_the_subgraph_output() -> None:
    """For {YAL001C, YAL004W}: gene_map = cumsum(mask) - 1 sends the Lazy edges under
    mask to exactly Subgraph's edge_index src [0, 0, 1, 2] dst [1, 0, 1, 2]; ~mask is
    Subgraph's pert_mask; node_ids under mask are Subgraph's node_ids. rmr: Lazy's
    reaction mask [F, F, T] gives reaction_map {2: 0}, and the masked hyperedges remap
    to [[0, 0], [2, 0]]. gpr differs by design: Lazy's mask is per gene ([F, T, F], the
    edge (gene 1, r1) survives although r1 is invalid), Subgraph also requires the
    reaction, so Subgraph's kept set is lazy_mask & reaction_mask[reaction] = [F, F, F].
    """
    cell_graph = _cell_graph()
    lazy = LazySubgraphRepresentation().process(cell_graph, PHENOTYPES, DOUBLE)
    sub = SubgraphRepresentation().process(cell_graph, PHENOTYPES, DOUBLE)
    gene_mask = lazy["gene"].mask
    gene_map = torch.cumsum(gene_mask.long(), dim=0) - 1
    kept = lazy[PHYSICAL].edge_index[:, lazy[PHYSICAL].mask]
    assert gene_map[kept].tolist() == [[0, 0, 1, 2], [1, 0, 1, 2]]
    assert gene_map[kept].tolist() == sub[PHYSICAL].edge_index.tolist()
    assert (~lazy[PHYSICAL].mask).tolist() == sub[PHYSICAL].pert_mask.tolist()
    assert [
        n for n, m in zip(lazy["gene"].node_ids, gene_mask.tolist(), strict=True) if m
    ] == sub["gene"].node_ids
    reaction_mask = lazy["reaction"].mask
    assert reaction_mask.tolist() == [False, False, True]
    reaction_map = torch.cumsum(reaction_mask.long(), dim=0) - 1
    rmr_kept = lazy[RMR].hyperedge_index[:, lazy[RMR].mask]
    remapped = torch.stack([reaction_map[rmr_kept[0]], rmr_kept[1]])
    assert remapped.tolist() == [[0, 0], [2, 0]]
    assert remapped.tolist() == sub[RMR].hyperedge_index.tolist()
    assert lazy[GPR].mask.tolist() == [False, True, False]
    gpr_reactions = lazy[GPR].hyperedge_index[1]
    sub_kept_gpr = lazy[GPR].mask & reaction_mask[gpr_reactions]
    assert sub_kept_gpr.tolist() == [False, False, False]
    assert sub_kept_gpr.tolist() == (~sub[GPR].pert_mask).tolist()


def test_neighbor_process_fails_because_perturbed_indices_is_a_list() -> None:
    """Finding: NeighborSubgraphRepresentation._process_gene_info returns
    perturbed_indices as a Python list (graph_processor.py:2481) and
    _build_khop_subgraph calls .tolist() on it (2491), so process() raises
    AttributeError on every input with at least one perturbed gene.
    """
    with pytest.raises(AttributeError, match="'list' object has no attribute 'tolist'"):
        NeighborSubgraphRepresentation(num_hops=1).process(
            _cell_graph(), PHENOTYPES, [_record(["YAL003W"], 0.9)]
        )


def _neighbor_pieces(num_hops: int) -> HeteroData:
    """Run the steps process() would run, with perturbed_indices patched to a tensor."""
    cell_graph = _cell_graph()
    data = [_record(["YAL003W"], 0.9)]
    processor = NeighborSubgraphRepresentation(num_hops=num_hops)
    processor._initialize_masks(cell_graph)
    gene_info = processor._process_gene_info(cell_graph, data)
    assert gene_info["perturbed_indices"] == [2]
    gene_info["perturbed_indices"] = torch.tensor([2])
    subgraph_info = processor._build_khop_subgraph(cell_graph, gene_info)
    out = HeteroData()
    processor._add_gene_data(out, cell_graph, gene_info, subgraph_info)
    for et, edge_data in subgraph_info["edge_data"].items():
        out[et].edge_index = edge_data["edge_index"]
        out[et].num_edges = edge_data["num_edges"]
    processor._process_metabolism(out, cell_graph, subgraph_info)
    processor._add_phenotype_data(out, PHENOTYPES, data)
    return out


def test_neighbor_one_hop_around_gene_2_adds_only_the_edge_source() -> None:
    """Finding: k_hop_subgraph (flow source_to_target) adds a node when it is the SOURCE
    of an edge whose target is already in the set. to_cell_data writes each undirected
    edge once, lower index -> higher, so from gene 2 the edges with target 2 are (1, 2)
    and (2, 2): one hop is {1, 2}, and gene 3 (target of (2, 3)) is never reached.
    Induced edges in original indices are positions 1, 5, 6: src [1, 1, 2] dst [2, 1, 2].
    Only r1 has a gene in {1, 2} (gene 1), so the gpr mask is [F, T, F] and the rmr mask
    keeps r1's two edges.
    """
    out = _neighbor_pieces(num_hops=1)
    gene = out["gene"]
    assert gene.node_ids == ["YAL002W", "YAL003W"]
    assert gene.num_nodes == 2
    assert gene.ids_pert == ["YAL003W"]
    assert gene.pert_mask.tolist() == [False, True]
    assert gene.x.tolist() == [[], []]
    assert gene.x_pert.tolist() == [[], []]
    assert out[PHYSICAL].edge_index.tolist() == [[1, 1, 2], [2, 1, 2]]
    assert out[PHYSICAL].num_edges == 3
    assert out[GPR].mask.tolist() == [False, True, False]
    assert out[RMR].mask.tolist() == [True, True, False, False, False, False]
    torch.testing.assert_close(gene.phenotype_values, torch.tensor([0.9]))
    assert gene.phenotype_sample_indices.tolist() == [0]


def test_neighbor_two_hops_around_gene_2_reach_gene_0() -> None:
    """Second hop: edges with target in {1, 2} add source 0 via (0, 1), so the set is
    {0, 1, 2} and the induced edges are positions 0, 1, 4, 5, 6: src [0, 1, 0, 1, 2]
    dst [1, 2, 0, 1, 2]. Both of r1's genes are now in the set: gpr mask [T, T, F].
    """
    out = _neighbor_pieces(num_hops=2)
    assert out["gene"].node_ids == ["YAL001C", "YAL002W", "YAL003W"]
    assert out["gene"].pert_mask.tolist() == [False, False, True]
    assert out[PHYSICAL].edge_index.tolist() == [[0, 1, 0, 1, 2], [1, 2, 0, 1, 2]]
    assert out[GPR].mask.tolist() == [True, True, False]
    assert out[RMR].mask.tolist() == [True, True, False, False, False, False]


def test_neighbor_metabolism_filters_reaction_nodes_but_not_hyperedge_indices() -> None:
    """Finding: _process_metabolism (graph_processor.py:2557-2583) stores only the
    reactions reached by the gene set (node_ids ['r1'], num_nodes 1) yet copies the FULL
    gpr and rmr hyperedge_index, whose reaction column still runs to index 2, so the
    edge tensors index a reaction store of size 1.
    """
    out = _neighbor_pieces(num_hops=1)
    assert out["reaction"].node_ids == ["r1"]
    assert out["reaction"].num_nodes == 1
    assert out["reaction"].pert_mask.tolist() == [False]
    assert out["reaction"].mask.tolist() == [True]
    assert out[GPR].hyperedge_index.tolist() == [[0, 1, 3], [0, 0, 1]]
    assert out[RMR].hyperedge_index.tolist() == [[0, 0, 1, 1, 2, 2], [0, 1, 1, 2, 2, 0]]
    assert out[RMR].num_edges == 6
    assert out["metabolite"].node_ids == ["m_a", "m_b", "m_c"]
    assert out["metabolite"].mask.tolist() == [True, True, True]
