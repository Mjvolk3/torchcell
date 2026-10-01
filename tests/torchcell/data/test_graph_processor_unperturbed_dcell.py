# tests/torchcell/data/test_graph_processor_unperturbed_dcell.py
# [[tests.torchcell.data.test_graph_processor_unperturbed_dcell]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/data/test_graph_processor_unperturbed_dcell.py
"""``Unperturbed`` and ``DCellGraphProcessor`` on a five-gene graph with GO and metabolism.

Fixture. Genes YAL001C..YAL005C at indices 0..4, physical path 0-1-2-3-4 plus self loops
(edge_index src [0, 1, 2, 3, 0, 1, 2, 3, 4] dst [1, 2, 3, 4, 0, 1, 2, 3, 4]), the
bipartite block of ``test_graph_processor_subgraph.py`` (reactions r1, r2, r3 and
metabolites m_a, m_b, m_c under ("reaction", "rmr", "metabolite") and
("gene", "gpr", "reaction")), and a GO DAG with edges child -> parent GO:a -> GO:root,
GO:b -> GO:root. GO terms sort to GO:a 0, GO:b 1, GO:root 2; strata are root 0, a 1,
b 1. Gene sets: root {YAL001C, YAL002W, YAL003W}, a {YAL001C, YAL002W}, b {YAL003W},
inserted root first, so ``go_gene_strata_state`` rows [go_idx, gene_idx, stratum, state]
are

    [2, 0, 0, 1], [2, 1, 0, 1], [2, 2, 0, 1], [0, 0, 1, 1], [0, 1, 1, 1], [1, 2, 1, 1]

DCell zeroes the state of every row whose gene_idx is perturbed: perturbing genes 0 and
3 zeroes rows 0 and 3 (gene 3 is annotated nowhere). Unperturbed copies the gene store
and the physical edges, stores each phenotype field as its own tensor, and copies the
metabolite nodes without the reaction block (see the Finding).
"""

from typing import Any

import networkx as nx
import pytest
import torch
from sortedcontainers import SortedDict

from torchcell.data.cell_data import to_cell_data
from torchcell.data.graph_processor import DCellGraphProcessor, Unperturbed
from torchcell.data.hetero_data import HeteroData
from torchcell.datamodels.schema import (
    Environment,
    FitnessExperiment,
    FitnessExperimentReference,
    FitnessPhenotype,
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
FULL_SRC = [0, 1, 2, 3, 0, 1, 2, 3, 4]
FULL_DST = [1, 2, 3, 4, 0, 1, 2, 3, 4]
BASE_STATE = [
    [2.0, 0.0, 0.0, 1.0],
    [2.0, 1.0, 0.0, 1.0],
    [2.0, 2.0, 0.0, 1.0],
    [0.0, 0.0, 1.0, 1.0],
    [0.0, 1.0, 1.0, 1.0],
    [1.0, 2.0, 1.0, 1.0],
]


def _bipartite() -> nx.Graph:
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


def _go() -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_node("GO:root", gene_set=["YAL001C", "YAL002W", "YAL003W"])
    graph.add_node("GO:a", gene_set=["YAL001C", "YAL002W"])
    graph.add_node("GO:b", gene_set=["YAL003W"])
    graph.add_edge("GO:a", "GO:root")
    graph.add_edge("GO:b", "GO:root")
    return graph


def _cell_graph(with_go: bool = True) -> HeteroData:
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
    incidence: dict[str, Any] = {"metabolism_bipartite": _bipartite()}
    if with_go:
        incidence["gene_ontology"] = _go()
    cell_graph = to_cell_data(multigraph, incidence_graphs=incidence)
    assert cell_graph[PHYSICAL].edge_index.tolist() == [FULL_SRC, FULL_DST]
    if with_go:
        assert cell_graph["gene_ontology"].node_ids == ["GO:a", "GO:b", "GO:root"]
        assert cell_graph["gene_ontology"].go_gene_strata_state.tolist() == BASE_STATE
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


DOUBLE = [_record(["YAL001C"], 0.9, se=0.05), _record(["YAL004W"], 0.4)]


def test_unperturbed_copies_the_gene_store_and_stores_each_phenotype_field() -> None:
    """X is the cell graph's tensor (identity), node_ids and the nine physical edges are
    copied unchanged, perturbation_indices are {0, 3}, and each field of
    FitnessPhenotype becomes its own tensor: fitness [0.9, 0.4], fitness_se [0.05]
    (record 1 has none, so only one value). The GO edge types are not copied.
    """
    cell_graph = _cell_graph()
    out = Unperturbed().process(cell_graph, PHENOTYPES, DOUBLE)
    gene = out["gene"]
    assert gene.x is cell_graph["gene"].x
    assert gene.node_ids == GENE_NAMES
    assert gene.num_nodes == 5
    assert sorted(gene.perturbed_genes) == ["YAL001C", "YAL004W"]
    assert sorted(gene.perturbation_indices.tolist()) == [0, 3]
    torch.testing.assert_close(gene.fitness, torch.tensor([0.9, 0.4]))
    torch.testing.assert_close(gene.fitness_se, torch.tensor([0.05]))
    assert out.edge_types == [PHYSICAL]
    assert out[PHYSICAL].edge_index.tolist() == [FULL_SRC, FULL_DST]
    assert out[PHYSICAL].num_edges == 9


def test_unperturbed_fills_an_absent_statistic_with_a_nan_placeholder() -> None:
    """One record without fitness_se: the field is written as [nan], fitness as [0.7]."""
    out = Unperturbed().process(_cell_graph(), PHENOTYPES, [_record(["YAL002W"], 0.7)])
    assert out["gene"].perturbation_indices.tolist() == [1]
    torch.testing.assert_close(out["gene"].fitness, torch.tensor([0.7]))
    assert out["gene"].fitness_se.isnan().tolist() == [True]


def test_unperturbed_copies_metabolite_nodes_but_not_the_bipartite_reactions() -> None:
    """Finding: Unperturbed looks for the legacy ("metabolite", "reactions",
    "metabolite") hypergraph (graph_processor.py:1951), which to_cell_data no longer
    emits; with the bipartite block it copies the three metabolite nodes and nothing of
    the reaction store, gpr or rmr edges.
    """
    out = Unperturbed().process(_cell_graph(), PHENOTYPES, DOUBLE)
    assert out.node_types == ["gene", "metabolite"]
    assert out["metabolite"].node_ids == ["m_a", "m_b", "m_c"]
    assert out["metabolite"].num_nodes == 3
    assert out.edge_types == [PHYSICAL]


def test_unperturbed_rejects_an_empty_batch_and_a_record_without_a_reference() -> None:
    """Both are ValueErrors with fixed messages; the reference key is required even
    though nothing is read from it.
    """
    cell_graph = _cell_graph()
    with pytest.raises(ValueError, match="Data list is empty"):
        Unperturbed().process(cell_graph, PHENOTYPES, [])
    with pytest.raises(
        ValueError,
        match="must contain both 'experiment' and 'experiment_reference' keys",
    ):
        Unperturbed().process(
            cell_graph, PHENOTYPES, [{"experiment": DOUBLE[0]["experiment"]}]
        )


def test_dcell_zeroes_the_state_of_rows_annotated_to_perturbed_genes() -> None:
    """Genes 0 and 3 perturbed: rows 0 ([2, 0, ...]) and 3 ([0, 0, ...]) carry gene 0 and
    go to state 0; gene 3 has no annotation, so no other row changes. The gene store
    keeps all 5 genes and no per-sample perturbation_indices_batch is written; the
    phenotype COO carries fitness [0.9, 0.4] and the single fitness_se 0.05.
    """
    cell_graph = _cell_graph()
    out = DCellGraphProcessor().process(cell_graph, PHENOTYPES, DOUBLE)
    gene = out["gene"]
    assert gene.num_nodes == 5
    assert gene.node_ids == GENE_NAMES
    assert gene.x is cell_graph["gene"].x
    assert sorted(gene.perturbed_genes) == ["YAL001C", "YAL004W"]
    assert gene.perturbation_indices.tolist() == [0, 3]
    assert gene.pert_mask.tolist() == [True, False, False, True, False]
    assert "perturbation_indices_batch" not in gene
    go = out["gene_ontology"]
    assert go.num_nodes == 3
    assert go.node_ids == ["GO:a", "GO:b", "GO:root"]
    expected = [row[:3] + [0.0 if row[1] == 0.0 else 1.0] for row in BASE_STATE]
    assert expected[0][3] == 0.0 and expected[3][3] == 0.0
    assert go.go_gene_strata_state.tolist() == expected
    assert cell_graph["gene_ontology"].go_gene_strata_state.tolist() == BASE_STATE
    torch.testing.assert_close(gene.phenotype_values, torch.tensor([0.9, 0.4]))
    assert gene.phenotype_sample_indices.tolist() == [0, 1]
    torch.testing.assert_close(gene.phenotype_stat_values, torch.tensor([0.05]))
    assert gene.phenotype_stat_sample_indices.tolist() == [0]
    assert gene.phenotype_stat_types == ["fitness_se"]
    assert out.edge_types == []


def test_dcell_orders_perturbation_indices_by_node_and_writes_no_batch_vector() -> None:
    """Records {YAL002W} and {YAL001C, YAL003W} give perturbation_indices [0, 1, 2]
    (node order, the union over records). The record-order batch vector [0, 1, 1] that
    used to sit beside it (and mispaired position 0 with record 0) is no longer
    written (issue #527 review); ``follow_batch`` builds the sample-level vector at
    collate. All six GO rows carry gene 0, 1 or 2 and go to state 0.
    """
    data = [_record(["YAL002W"], 0.9), _record(["YAL001C", "YAL003W"], 0.4)]
    out = DCellGraphProcessor().process(_cell_graph(), PHENOTYPES, data)
    gene = out["gene"]
    assert gene.perturbation_indices.tolist() == [0, 1, 2]
    assert "perturbation_indices_batch" not in gene
    assert gene.pert_mask.tolist() == [True, True, True, False, False]
    state = out["gene_ontology"].go_gene_strata_state
    assert state[:, 3].tolist() == [0.0] * 6
    assert state[:, :3].tolist() == [row[:3] for row in BASE_STATE]


def test_dcell_without_a_go_block_returns_only_the_gene_store() -> None:
    """No gene_ontology node type in the cell graph: node_types is ['gene'] and the
    perturbation tensors are still written.
    """
    out = DCellGraphProcessor().process(
        _cell_graph(with_go=False), PHENOTYPES, [_record(["YAL005C"], 0.5)]
    )
    assert out.node_types == ["gene"]
    assert out["gene"].perturbation_indices.tolist() == [4]
    assert "perturbation_indices_batch" not in out["gene"]


def test_dcell_rejects_an_empty_batch() -> None:
    """No records is an error, not an unperturbed state tensor."""
    with pytest.raises(ValueError, match="Data list is empty"):
        DCellGraphProcessor().process(_cell_graph(), PHENOTYPES, [])
