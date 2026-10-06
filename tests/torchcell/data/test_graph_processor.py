# tests/torchcell/data/test_graph_processor.py
# [[tests.torchcell.data.test_graph_processor]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/data/test_graph_processor.py
"""The ``Perturbation`` graph processor on a three-gene cell graph, exact tensors.

Two fitness records perturb {YAL001C} and {YAL002W, YAL003W}. The processor stores the
UNION of perturbed genes for the whole batch, so ``perturbation_indices`` is [0, 1, 2]
and ``pert_mask`` is all True. Phenotypes are COO: one value per record for the
``fitness`` label, type index 0, sample indices 0 and 1; ``fitness_se`` is unset on both
records, so the statistic tensors are empty but the statistic name is still declared.

With two phenotype classes [FitnessPhenotype, GeneInteractionPhenotype] the type index
is the position in that list: a fitness record contributes (0.9, type 0) and a gene
interaction record (-0.2, type 1) with its p-value 0.01 at stat type 1; the reference
phenotypes (fitness 1.0, gene_interaction 0.0) never enter the tensors. A dict-valued
label (CalMorph) flattens in sorted-key order: {A101_A1B: 2.0, A101_A: 1.0} becomes
[1.0, 2.0] because "A101_A" < "A101_A1B".

``tests/torchcell/data/test_graph_processor_subgraph.py`` and
``test_graph_processor_unperturbed_dcell.py`` cover the other processors on synthetic
graphs; ``test_graph_processor_equivalence.py`` covers the subgraph processors on real
data behind ``--data``.

2026.09.30 (Phase 14): the branches those files left open, on a hand-built three-gene
graph (``_small_graph``: x = [[1], [2], [3]], physical edges (0, 1), (1, 2), (2, 2),
regulatory edge (2, 0); optionally three reactions with subsystem (Growth, Other,
Growth) and no gpr edges, and two metabolites with no rmr edges) and on the conftest
``dcell_graph`` (four genes, template rows [go, gene, stratum, state] =
[1, 0, 1, 1], [1, 1, 1, 1], [2, 2, 1, 1], [2, 3, 1, 1]). Expected values, each derived
in its test: perturbing gene 1 keeps genes [0, 2] with map {0: 0, 2: 1}, so the physical
survivor (2, 2) becomes (1, 1) and the regulatory (2, 0) becomes (1, 0); without gpr
edges every reaction is kept; the DCell union {0, 2} zeroes the state of rows (1, 0) and
(2, 2).

2026.09.30 (issue #527): the six Phase 14 Findings are fixed and asserted as
contracts. ``w_growth`` has one source, the stored cell-graph tensor, in all three
masking processors; a ``subsystem`` attribute (list or tensor) is no longer read; the
incidence cache is rebuilt when a different cell graph arrives; ``Unperturbed`` skips a
phenotype with no statistic name; DCell writes no per-sample
``perturbation_indices_batch`` (``follow_batch`` builds it at collate);
the neighbor processor writes the [NaN] placeholder.
"""

from types import SimpleNamespace
from typing import Any

import networkx as nx
import pytest
import torch
from pydantic import BaseModel
from sortedcontainers import SortedDict
from torch_geometric.data import Batch, HeteroData

from torchcell.data.cell_data import to_cell_data
from torchcell.data.graph_processor import (
    DCellGraphProcessor,
    IncidenceSubgraphRepresentation,
    LazySubgraphRepresentation,
    NeighborSubgraphRepresentation,
    Perturbation,
    SubgraphRepresentation,
    Unperturbed,
)
from torchcell.datamodels.schema import (
    CalMorphExperiment,
    CalMorphExperimentReference,
    CalMorphPhenotype,
    Environment,
    FitnessExperiment,
    FitnessExperimentReference,
    FitnessPhenotype,
    GeneInteractionExperiment,
    GeneInteractionExperimentReference,
    GeneInteractionPhenotype,
    Genotype,
    KanMxDeletionPerturbation,
    Media,
    ReferenceGenome,
    VisualScorePhenotype,
)
from torchcell.graph.graph import GeneGraph, GeneMultiGraph
from torchcell.sequence import GeneSet

GENES = GeneSet(["YAL001C", "YAL002W", "YAL003W"])
PHENOTYPES: list[Any] = [FitnessPhenotype]  # the processor takes the classes
TWO_PHENOTYPES: list[Any] = [FitnessPhenotype, GeneInteractionPhenotype]
INTERACTION_ONLY: list[Any] = [GeneInteractionPhenotype]
VISUAL_THEN_FITNESS: list[Any] = [VisualScorePhenotype, FitnessPhenotype]
VISUAL_THEN_INTERACTION: list[Any] = [VisualScorePhenotype, GeneInteractionPhenotype]
VISUAL_ONLY: list[Any] = [VisualScorePhenotype]
CALMORPH: list[Any] = [CalMorphPhenotype]
ENVIRONMENT = Environment(media=Media(name="YPD", state="solid", is_synthetic=False))
GENOME = ReferenceGenome(species="Saccharomyces cerevisiae", strain="S288C")


def _genotype(genes: list[str]) -> Genotype:
    perturbations = [
        KanMxDeletionPerturbation(systematic_gene_name=g, perturbed_gene_name=g)
        for g in genes
    ]
    return Genotype(perturbations=perturbations)  # type: ignore[arg-type]


def _cell_graph() -> HeteroData:
    """Three genes plus one physical edge, so a processor that copied edges would show it."""
    base = nx.Graph()
    base.add_nodes_from(GENES)
    physical = nx.Graph()
    physical.add_nodes_from(GENES)
    physical.add_edge("YAL001C", "YAL002W")
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
    cell_graph = to_cell_data(multigraph)
    assert cell_graph["gene", "physical_interaction", "gene"].num_edges == 4
    return cell_graph


def _record(
    genes: list[str], fitness: float, se: float | None = None
) -> dict[str, Any]:
    return {
        "experiment": FitnessExperiment(
            dataset_name="toy",
            genotype=_genotype(genes),
            environment=ENVIRONMENT,
            phenotype=FitnessPhenotype(fitness=fitness, fitness_se=se),
        ),
        "experiment_reference": FitnessExperimentReference(
            dataset_name="toy",
            genome_reference=GENOME,
            environment_reference=ENVIRONMENT,
            phenotype_reference=FitnessPhenotype(fitness=1.0),
        ),
    }


def _interaction_record(
    genes: list[str], score: float, p_value: float | None = None
) -> dict[str, Any]:
    return {
        "experiment": GeneInteractionExperiment(
            dataset_name="toy",
            genotype=_genotype(genes),
            environment=ENVIRONMENT,
            phenotype=GeneInteractionPhenotype(
                gene_interaction=score, gene_interaction_p_value=p_value
            ),
        ),
        "experiment_reference": GeneInteractionExperimentReference(
            dataset_name="toy",
            genome_reference=GENOME,
            environment_reference=ENVIRONMENT,
            phenotype_reference=GeneInteractionPhenotype(gene_interaction=0.0),
        ),
    }


def _calmorph_record(
    genes: list[str], calmorph: dict[str, float], cv: dict[str, float] | None
) -> dict[str, Any]:
    return {
        "experiment": CalMorphExperiment(
            dataset_name="toy",
            genotype=_genotype(genes),
            environment=ENVIRONMENT,
            phenotype=CalMorphPhenotype(
                calmorph=calmorph, calmorph_coefficient_of_variation=cv
            ),
        ),
        "experiment_reference": CalMorphExperimentReference(
            dataset_name="toy",
            genome_reference=GENOME,
            environment_reference=ENVIRONMENT,
            phenotype_reference=CalMorphPhenotype(calmorph={"A101_A": 0.0}),
        ),
    }


def test_perturbation_processor_unions_the_perturbed_genes() -> None:
    """{YAL001C} and {YAL002W, YAL003W} give indices [0, 1, 2] and an all-True pert_mask."""
    data = [_record(["YAL001C"], 0.9), _record(["YAL002W", "YAL003W"], 0.4)]
    out = Perturbation().process(_cell_graph(), PHENOTYPES, data)
    assert out["gene"].num_nodes == 3
    assert out["gene"].perturbation_indices.tolist() == [0, 1, 2]
    assert out["gene"].pert_mask.tolist() == [True, True, True]
    assert out["gene"].mask.tolist() == [False, False, False]
    assert sorted(out["gene"].perturbed_genes) == ["YAL001C", "YAL002W", "YAL003W"]
    assert out.edge_types == []  # the four physical edges of the input are not copied
    assert out.node_types == ["gene"]


def test_perturbation_processor_stores_phenotypes_in_coo_form() -> None:
    """Values [0.9, 0.4] at type 0, samples [0, 1]; the se statistic is declared but empty."""
    data = [_record(["YAL001C"], 0.9), _record(["YAL003W"], 0.4)]
    out = Perturbation().process(_cell_graph(), PHENOTYPES, data)
    gene = out["gene"]
    torch.testing.assert_close(gene.phenotype_values, torch.tensor([0.9, 0.4]))
    assert gene.phenotype_type_indices.tolist() == [0, 0]
    assert gene.phenotype_sample_indices.tolist() == [0, 1]
    assert gene.phenotype_types == ["fitness"]
    assert gene.phenotype_stat_values.numel() == 0
    assert gene.phenotype_stat_types == ["fitness_se"]
    assert gene.perturbation_indices.tolist() == [0, 2]


def test_perturbation_processor_records_a_statistic_when_present() -> None:
    """A record with fitness_se 0.05 fills the stat tensors for that sample only."""
    data = [_record(["YAL001C"], 0.9, se=0.05), _record(["YAL002W"], 0.4)]
    out = Perturbation().process(_cell_graph(), PHENOTYPES, data)
    gene = out["gene"]
    torch.testing.assert_close(gene.phenotype_stat_values, torch.tensor([0.05]))
    assert gene.phenotype_stat_type_indices.tolist() == [0]
    assert gene.phenotype_stat_sample_indices.tolist() == [0]


def test_perturbation_processor_indexes_types_by_position_in_phenotype_info() -> None:
    """[FitnessPhenotype, GeneInteractionPhenotype]: the fitness record fills (0.9, type
    0, sample 0), the interaction record (-0.2, type 1, sample 1); the p-value 0.01 is
    the only statistic, at stat type 1 (position of gene_interaction_p_value in
    [fitness_se, gene_interaction_p_value]). Neither reference phenotype (fitness 1.0,
    gene_interaction 0.0) appears anywhere.
    """
    data = [_record(["YAL001C"], 0.9), _interaction_record(["YAL002W"], -0.2, 0.01)]
    out = Perturbation().process(_cell_graph(), TWO_PHENOTYPES, data)
    gene = out["gene"]
    torch.testing.assert_close(gene.phenotype_values, torch.tensor([0.9, -0.2]))
    assert gene.phenotype_type_indices.tolist() == [0, 1]
    assert gene.phenotype_sample_indices.tolist() == [0, 1]
    assert gene.phenotype_types == ["fitness", "gene_interaction"]
    torch.testing.assert_close(gene.phenotype_stat_values, torch.tensor([0.01]))
    assert gene.phenotype_stat_type_indices.tolist() == [1]
    assert gene.phenotype_stat_sample_indices.tolist() == [1]
    assert gene.phenotype_stat_types == ["fitness_se", "gene_interaction_p_value"]
    assert gene.perturbation_indices.tolist() == [0, 1]


def test_perturbation_processor_flattens_dict_labels_in_sorted_key_order() -> None:
    """CalMorph {A101_A1B: 2.0, A101_A: 1.0} (insertion order) is stored as [1.0, 2.0]
    because the keys sort A101_A < A101_A1B; both values share type 0 and sample 0. The
    CV statistic {ACV101_A1B: 0.2, ACV101_A: 0.1} flattens the same way to [0.1, 0.2].
    """
    data = [
        _calmorph_record(
            ["YAL003W"],
            {"A101_A1B": 2.0, "A101_A": 1.0},
            {"ACV101_A1B": 0.2, "ACV101_A": 0.1},
        )
    ]
    out = Perturbation().process(_cell_graph(), CALMORPH, data)
    gene = out["gene"]
    torch.testing.assert_close(gene.phenotype_values, torch.tensor([1.0, 2.0]))
    assert gene.phenotype_type_indices.tolist() == [0, 0]
    assert gene.phenotype_sample_indices.tolist() == [0, 0]
    assert gene.phenotype_types == ["calmorph"]
    torch.testing.assert_close(gene.phenotype_stat_values, torch.tensor([0.1, 0.2]))
    assert gene.phenotype_stat_type_indices.tolist() == [0, 0]
    assert gene.phenotype_stat_sample_indices.tolist() == [0, 0]
    assert gene.phenotype_stat_types == ["calmorph_coefficient_of_variation"]


def test_perturbation_processor_rejects_an_empty_batch() -> None:
    """No records is an error, not an empty graph."""
    with pytest.raises(ValueError, match="Data list is empty"):
        Perturbation().process(_cell_graph(), PHENOTYPES, [])


# ---------------------------------------------------------------------------
# 2026.09.30 (Phase 14): the branches the files above left open.

PHYSICAL = ("gene", "physical_interaction", "gene")
REGULATORY = ("gene", "regulatory_interaction", "gene")
GPR = ("gene", "gpr", "reaction")
METABOLITE_HYPEREDGE = ("metabolite", "reactions", "metabolite")
SUBGRAPH_PROCESSORS = [SubgraphRepresentation, IncidenceSubgraphRepresentation]
MASKING_PROCESSORS = [
    SubgraphRepresentation,
    IncidenceSubgraphRepresentation,
    LazySubgraphRepresentation,
]


def _small_graph(reactions: bool = False, metabolites: bool = False) -> HeteroData:
    """Three genes, x = [[1], [2], [3]]; physical edges (0, 1), (1, 2), (2, 2) at
    positions 0, 1, 2; one regulatory edge (2, 0). Optionally three reactions with a
    ``subsystem`` tuple (Growth, Other, Growth) and NO gpr edges, and two metabolites
    with NO rmr edges.
    """
    graph = HeteroData()
    graph["gene"].node_ids = ["YAL001C", "YAL002W", "YAL003W"]
    graph["gene"].num_nodes = 3
    graph["gene"].x = torch.tensor([[1.0], [2.0], [3.0]])
    graph[PHYSICAL].edge_index = torch.tensor([[0, 1, 2], [1, 2, 2]])
    graph[REGULATORY].edge_index = torch.tensor([[2], [0]])
    if reactions:
        graph["reaction"].num_nodes = 3
        graph["reaction"].subsystem = ("Growth", "Other", "Growth")
    if metabolites:
        graph["metabolite"].num_nodes = 2
        graph["metabolite"].node_ids = ["m0", "m1"]
    return graph


def _plain(store: Any) -> dict[str, Any]:
    """A store as plain Python, with ids_pert and perturbed_genes sorted (from sets)."""
    items = {k: (v.tolist() if torch.is_tensor(v) else v) for k, v in store.items()}
    for key in ("ids_pert", "perturbed_genes"):
        if key in items:
            items[key] = sorted(items[key])
    return items


@pytest.mark.parametrize(
    "processor", SUBGRAPH_PROCESSORS, ids=["subgraph", "incidence"]
)
def test_subgraph_processors_on_a_gene_only_graph(processor: Any) -> None:
    """Perturbing YAL002W (gene 1) keeps genes [0, 2], gene map {0: 0, 2: 1}.

    Physical: (0, 1) and (1, 2) touch gene 1 and go; (2, 2) relabels to (1, 1).
    Regulatory: gene 1 has no regulatory edge, so (2, 0) survives as (1, 0) and the
    incidence path's "nothing to remove" branch runs. With no reaction or metabolite
    node type the output has only the gene store and no reaction or metabolite mask.
    """
    out = processor().process(_small_graph(), PHENOTYPES, [_record(["YAL002W"], 0.9)])
    assert out.node_types == ["gene"]
    assert out.edge_types == [PHYSICAL, REGULATORY]
    assert _plain(out["gene"]) == {
        "node_ids": ["YAL001C", "YAL003W"],
        "num_nodes": 2,
        "ids_pert": ["YAL002W"],
        "perturbation_indices": [1],
        "x": [[1.0], [3.0]],
        "x_pert": [[2.0]],
        "phenotype_values": [pytest.approx(0.9)],
        "phenotype_type_indices": [0],
        "phenotype_sample_indices": [0],
        "phenotype_types": ["fitness"],
        "pert_mask": [False, True, False],
    }
    assert _plain(out[PHYSICAL]) == {
        "edge_index": [[1], [1]],
        "num_edges": 1,
        "pert_mask": [True, True, False],
    }
    assert _plain(out[REGULATORY]) == {
        "edge_index": [[1], [0]],
        "num_edges": 1,
        "pert_mask": [False],
    }


def test_the_lazy_processor_on_a_gene_only_graph_masks_instead_of_cutting() -> None:
    """Same input: every node kept, the physical mask is [F, F, T] (the two edges that
    touch gene 1), the regulatory mask [T], and no reaction or metabolite store.
    """
    out = LazySubgraphRepresentation().process(
        _small_graph(), PHENOTYPES, [_record(["YAL002W"], 0.9)]
    )
    assert out.node_types == ["gene"]
    assert out["gene"].pert_mask.tolist() == [False, True, False]
    assert out["gene"].mask.tolist() == [True, False, True]
    assert out[PHYSICAL].edge_index.tolist() == [[0, 1, 2], [1, 2, 2]]
    assert out[PHYSICAL].mask.tolist() == [False, False, True]
    assert out[REGULATORY].mask.tolist() == [True]


@pytest.mark.parametrize("processor", MASKING_PROCESSORS, ids=["sub", "inc", "lazy"])
def test_reactions_without_gpr_edges_are_all_kept(processor: Any) -> None:
    """With no gpr edge type every reaction is valid: node_ids [0, 1, 2], an all-False
    pert_mask and no gpr store. The metabolites have no rmr edges, so their output store
    carries only the masks (no node_ids, no num_nodes).
    """
    out = processor().process(
        _small_graph(reactions=True, metabolites=True),
        PHENOTYPES,
        [_record(["YAL002W"], 0.9)],
    )
    assert GPR not in out.edge_types
    reaction = _plain(out["reaction"])
    assert reaction["num_nodes"] == 3
    assert reaction["node_ids"] == [0, 1, 2]
    assert reaction["pert_mask"] == [False, False, False]
    assert "w_growth" not in reaction
    metabolite = _plain(out["metabolite"])
    assert metabolite.pop("pert_mask") == [False, False]
    if issubclass(processor, LazySubgraphRepresentation):
        assert reaction["mask"] == [True, True, True]
        assert metabolite.pop("mask") == [True, True]
    assert metabolite == {}


@pytest.mark.parametrize("processor", MASKING_PROCESSORS, ids=["sub", "inc", "lazy"])
def test_every_masking_processor_returns_the_stored_w_growth(processor: Any) -> None:
    """Contract (issue #527): ``w_growth`` has ONE source, the tensor ``to_cell_data``
    stores on the cell graph (1 for a Growth-subsystem reaction), subset to the kept
    reactions. The graph carries w_growth [0.5, 0, 1] and a contradicting subsystem
    (Growth, Other, Growth); without gpr edges every reaction is kept, so all three
    processors return exactly [0.5, 0, 1]. Before the fix Lazy recomputed [1, 0, 1]
    from the subsystem labels, and on a real ``to_cell_data`` graph (which carries no
    ``subsystem``) returned all zeros.
    """
    graph = _small_graph(reactions=True)
    graph["reaction"].w_growth = torch.tensor([0.5, 0.0, 1.0])
    out = processor().process(graph, PHENOTYPES, [_record(["YAL002W"], 0.9)])
    assert out["reaction"].w_growth.tolist() == [0.5, 0.0, 1.0]


def test_a_tensor_subsystem_does_not_touch_the_stored_w_growth() -> None:
    """Contract (issue #527): a tensor-valued ``subsystem`` is not decoded at all; the
    stored w_growth [0, 1, 0], the opposite of any "1 = Growth" reading of [1, 0, 1], is
    returned exactly. Before the fix the tensor branch
    compared each 0-d tensor with the string "Growth" and always produced [0, 0, 0].
    """
    graph = _small_graph(reactions=True)
    graph["reaction"].subsystem = torch.tensor([1, 0, 1])
    graph["reaction"].w_growth = torch.tensor([0.0, 1.0, 0.0])
    out = LazySubgraphRepresentation().process(
        graph, PHENOTYPES, [_record(["YAL002W"], 0.9)]
    )
    assert out["reaction"].w_growth.tolist() == [0.0, 1.0, 0.0]


@pytest.mark.parametrize("processor", MASKING_PROCESSORS, ids=["sub", "inc", "lazy"])
def test_statistic_types_index_the_statistic_list_not_the_label_list(
    processor: Any,
) -> None:
    """[VisualScorePhenotype, FitnessPhenotype]: visual_score has no statistic name, so
    the statistic list is [fitness_se] alone. The fitness value is at label type 1 and
    its se at statistic type 0; a consumer must index each through its own list.
    """
    out = processor().process(
        _small_graph(), VISUAL_THEN_FITNESS, [_record(["YAL002W"], 0.9, se=0.05)]
    )
    gene = out["gene"]
    assert gene.phenotype_types == ["visual_score", "fitness"]
    assert gene.phenotype_type_indices.tolist() == [1]
    assert gene.phenotype_stat_types == ["fitness_se"]
    assert gene.phenotype_stat_type_indices.tolist() == [0]
    torch.testing.assert_close(gene.phenotype_stat_values, torch.tensor([0.05]))


@pytest.mark.parametrize("processor", MASKING_PROCESSORS, ids=["sub", "inc", "lazy"])
def test_a_label_no_record_carries_becomes_a_nan_placeholder(processor: Any) -> None:
    """Asking for gene_interaction on a fitness record: values [NaN], type [0], sample
    [0], and none of the four statistic keys (the p-value is absent too).
    """
    out = processor().process(
        _small_graph(), INTERACTION_ONLY, [_record(["YAL002W"], 0.9, 0.05)]
    )
    gene = out["gene"]
    assert torch.isnan(gene.phenotype_values).tolist() == [True]
    assert gene.phenotype_type_indices.tolist() == [0]
    assert gene.phenotype_sample_indices.tolist() == [0]
    assert gene.phenotype_types == ["gene_interaction"]
    assert not any(key.startswith("phenotype_stat") for key in gene.keys())


def test_incidence_reuses_a_built_cache_across_samples() -> None:
    """``build_cache`` first, then two samples on one processor: each output is exact.
    Gene 0 removed: physical keeps (1, 2), (2, 2) -> [[0, 1], [1, 1]]; the regulatory
    edge (2, 0) goes, leaving an empty [2, 0] index.
    """
    processor = IncidenceSubgraphRepresentation()
    info = processor.build_cache(_small_graph())
    # incidence lists: physical gene 0 [0], gene 1 [0, 1], gene 2 [1, 2] (the self
    # loop once) and regulatory gene 0 [0], gene 2 [0], so (1 + 2 + 2 + 1 + 1) // 2 = 3
    # for 4 edges (the self-loop undercount Finding in the subgraph file)
    assert (info["num_edge_types"], info["total_edges"]) == (2, 3)
    first = processor.process(_small_graph(), PHENOTYPES, [_record(["YAL001C"], 0.9)])
    assert first[PHYSICAL].edge_index.tolist() == [[0, 1], [1, 1]]
    assert first[REGULATORY].edge_index.shape == (2, 0)
    assert first[REGULATORY].pert_mask.tolist() == [True]
    second = processor.process(_small_graph(), PHENOTYPES, [_record(["YAL002W"], 0.9)])
    assert second[PHYSICAL].edge_index.tolist() == [[1], [1]]
    assert second[REGULATORY].edge_index.tolist() == [[1], [0]]


def _four_gene_graph() -> HeteroData:
    """Four genes (one more than ``_small_graph``); physical edges (1, 2), (2, 2),
    (0, 1), (3, 0) at positions 0..3; one regulatory edge (3, 2).
    """
    graph = HeteroData()
    graph["gene"].node_ids = ["YAL001C", "YAL002W", "YAL003W", "YAL004W"]
    graph["gene"].num_nodes = 4
    graph["gene"].x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    graph[PHYSICAL].edge_index = torch.tensor([[1, 2, 0, 3], [2, 2, 1, 0]])
    graph[REGULATORY].edge_index = torch.tensor([[3], [2]])
    return graph


def test_a_cached_processor_rebuilds_its_cache_for_a_different_graph() -> None:
    """Contract (issue #527): the incidence cache is tied to the gene count and the
    gene-gene ``edge_index`` tensors it was built from, and is REBUILT when a different
    graph arrives. Built on ``_small_graph`` (3 genes), then fed the four-gene graph:
    removing gene 0 drops positions 2 and 3, the kept genes [1, 2, 3] relabel to
    {1: 0, 2: 1, 3: 2}, so physical (1, 2), (2, 2) become [[0, 1], [1, 1]] with
    pert_mask [F, F, T, T], and regulatory (3, 2) becomes [[2], [1]]. The Lazy mask is
    [T, T, F, F]. Before the fix the stale three-gene cache dropped the wrong position
    and emitted node index -1. Returning to the first graph rebuilds again.
    """
    data = [_record(["YAL001C"], 0.9)]
    fresh = SubgraphRepresentation().process(_four_gene_graph(), PHENOTYPES, data)
    assert fresh[PHYSICAL].edge_index.tolist() == [[0, 1], [1, 1]]

    incidence = IncidenceSubgraphRepresentation()
    incidence.build_cache(_small_graph())
    second = incidence.process(_four_gene_graph(), PHENOTYPES, data)
    assert second["gene"].node_ids == ["YAL002W", "YAL003W", "YAL004W"]
    assert second[PHYSICAL].edge_index.tolist() == [[0, 1], [1, 1]]
    assert second[PHYSICAL].pert_mask.tolist() == [False, False, True, True]
    assert second[REGULATORY].edge_index.tolist() == [[2], [1]]
    assert [len(e) for e in incidence.edge_incidence_cache[PHYSICAL]] == [2, 2, 2, 1]
    back = incidence.process(_small_graph(), PHENOTYPES, data)
    assert back[PHYSICAL].edge_index.tolist() == [[0, 1], [1, 1]]
    assert [len(e) for e in incidence.edge_incidence_cache[PHYSICAL]] == [1, 2, 2]

    lazy = LazySubgraphRepresentation()
    lazy.build_cache(_small_graph())
    assert lazy.process(_four_gene_graph(), PHENOTYPES, data)[
        PHYSICAL
    ].mask.tolist() == [True, True, False, False]


@pytest.mark.parametrize(
    "processor",
    [IncidenceSubgraphRepresentation, LazySubgraphRepresentation],
    ids=["incidence", "lazy"],
)
def test_the_cache_is_rebuilt_for_same_size_graphs_with_different_edges(
    processor: Any,
) -> None:
    """A graph with the same gene count and edge count but different edges is a
    different graph: ``build_cache`` on it reports a fresh build (3 incidence pairs,
    as in ``test_incidence_reuses_a_built_cache_across_samples``) and gene 0's list
    becomes [2] (the edge (0, 1) now at position 2). An equal-content copy of the first
    graph is recognized: ``build_cache`` returns the zero report.
    """
    instance = processor()
    instance.build_cache(_small_graph())
    assert instance.build_cache(_small_graph()) == {
        "total_time_ms": 0.0,
        "num_edge_types": 0,
        "total_edges": 0,
    }
    permuted = _small_graph()
    permuted[PHYSICAL].edge_index = torch.tensor([[1, 2, 0], [2, 2, 1]])
    info = instance.build_cache(permuted)
    assert (info["num_edge_types"], info["total_edges"]) == (2, 3)
    assert instance.edge_incidence_cache[PHYSICAL][0].tolist() == [2]


def _unperturbed_graph() -> HeteroData:
    """The small graph plus a coexpression edge type and a metabolite hypergraph whose
    reaction 0 names YAL003W (gene 2) and YBR001C (not a node).
    """
    graph = _small_graph()
    graph["gene", "coexpression", "gene"].edge_index = torch.tensor([[0], [2]])
    graph["metabolite"].num_nodes = 2
    graph["metabolite"].node_ids = ["m0", "m1"]
    graph[METABOLITE_HYPEREDGE].hyperedge_index = torch.tensor([[0, 1], [0, 0]])
    graph[METABOLITE_HYPEREDGE].stoichiometry = torch.tensor([-1.0, 1.0])
    graph[METABOLITE_HYPEREDGE].num_edges = 1
    graph[METABOLITE_HYPEREDGE].reaction_to_genes = {0: ["YAL003W", "YBR001C"], 1: []}
    return graph


def test_unperturbed_copies_named_edge_types_and_maps_reaction_genes() -> None:
    """Physical and regulatory edges are copied, coexpression is not; the metabolite
    hypergraph is copied and its reaction_to_genes becomes indices, -1 for a gene
    that is not a node: {0: [2, -1], 1: []}. Fitness 0.9 is stored as a field and the
    absent se as [NaN].
    """
    out = Unperturbed().process(
        _unperturbed_graph(), PHENOTYPES, [_record(["YAL002W"], 0.9)]
    )
    assert out.node_types == ["gene", "metabolite"]
    assert out.edge_types == [PHYSICAL, REGULATORY, METABOLITE_HYPEREDGE]
    assert out[PHYSICAL].edge_index.tolist() == [[0, 1, 2], [1, 2, 2]]
    hyper = out[METABOLITE_HYPEREDGE]
    assert hyper.hyperedge_index.tolist() == [[0, 1], [0, 0]]
    assert hyper.stoichiometry.tolist() == [-1.0, 1.0]
    assert hyper.num_edges == 1
    assert hyper.reaction_to_genes_indices == {0: [2, -1], 1: []}
    assert out["metabolite"].node_ids == ["m0", "m1"]
    gene = out["gene"]
    assert gene.perturbation_indices.tolist() == [1]
    assert gene.fitness.tolist() == [pytest.approx(0.9)]
    assert torch.isnan(gene.fitness_se).tolist() == [True]


def test_unperturbed_on_a_gene_only_graph_writes_only_the_gene_store() -> None:
    """No metabolite node type: the output is the gene store and the two copied edge
    types, and perturbation_indices is the node position of YAL003W, [2].
    """
    out = Unperturbed().process(_small_graph(), PHENOTYPES, [_record(["YAL003W"], 0.4)])
    assert out.node_types == ["gene"]
    assert out.edge_types == [PHYSICAL, REGULATORY]
    assert out["gene"].perturbation_indices.tolist() == [2]
    assert out["gene"].perturbed_genes == ["YAL003W"]


def test_the_lazy_cache_property_refuses_before_a_build() -> None:
    """Reading the cache before ``process`` raises the exact message of line 1293."""
    with pytest.raises(
        RuntimeError,
        match=r"^Incidence cache not built\. Call _build_incidence_cache\(\) first\.$",
    ):
        LazySubgraphRepresentation().edge_incidence_cache  # noqa: B018


def test_unperturbed_refuses_a_gene_that_is_not_a_node() -> None:
    """A perturbed gene absent from the graph fails the node lookup with the list error."""
    with pytest.raises(ValueError, match=r"^'YBR001C' is not in list$"):
        Unperturbed().process(_small_graph(), PHENOTYPES, [_record(["YBR001C"], 0.9)])


def test_unperturbed_skips_a_phenotype_without_a_statistic() -> None:
    """Contract (issue #527): ``Unperturbed`` writes the statistic field only when the
    phenotype class names one. VisualScorePhenotype has none, so the gene store gains
    only ``visual_score`` (the fitness record lacks it: [NaN]); FitnessPhenotype names
    ``fitness_se``, so both fields are written. Before the fix the None statistic name
    reached ``getattr`` and raised TypeError.
    """
    base = {"x", "node_ids", "num_nodes", "perturbed_genes", "perturbation_indices"}
    visual = Unperturbed().process(
        _small_graph(), VISUAL_ONLY, [_record(["YAL002W"], 0.9)]
    )["gene"]
    assert set(visual.keys()) - base == {"visual_score"}
    assert torch.isnan(visual.visual_score).tolist() == [True]
    fitness = Unperturbed().process(
        _small_graph(), PHENOTYPES, [_record(["YAL002W"], 0.9, 0.05)]
    )["gene"]
    assert set(fitness.keys()) - base == {"fitness", "fitness_se"}
    torch.testing.assert_close(fitness.fitness_se, torch.tensor([0.05]))


class _ProfilePhenotype(BaseModel):
    """A stand-in phenotype whose label is a list and whose statistic is a tuple."""

    label_name: str = "profile"
    label_statistic_name: str = "profile_se"
    profile: list[float]
    profile_se: tuple[float, ...]


def test_perturbation_flattens_list_and_tuple_labels_in_order() -> None:
    """Profile [0.1, 0.2] and profile_se (0.01, 0.02, 0.03) enter in their own order,
    every entry at type 0 and sample 0.
    """
    genotype = Genotype(
        perturbations=[
            KanMxDeletionPerturbation(
                systematic_gene_name="YAL003W", perturbed_gene_name="YAL003W"
            )
        ]
    )
    item: dict[str, Any] = {
        "experiment": SimpleNamespace(
            genotype=genotype,
            phenotype=_ProfilePhenotype(
                profile=[0.1, 0.2], profile_se=(0.01, 0.02, 0.03)
            ),
        )
    }
    phenotype_info: list[Any] = [_ProfilePhenotype]
    gene = Perturbation().process(_small_graph(), phenotype_info, [item])["gene"]
    torch.testing.assert_close(gene.phenotype_values, torch.tensor([0.1, 0.2]))
    assert gene.phenotype_type_indices.tolist() == [0, 0]
    assert gene.phenotype_sample_indices.tolist() == [0, 0]
    torch.testing.assert_close(
        gene.phenotype_stat_values, torch.tensor([0.01, 0.02, 0.03])
    )
    assert gene.phenotype_stat_sample_indices.tolist() == [0, 0, 0]
    assert gene.phenotype_stat_types == ["profile_se"]
    assert gene.perturbation_indices.tolist() == [2]
    assert gene.mask.tolist() == [True, True, False]


def test_perturbation_placeholder_keeps_the_four_statistic_keys() -> None:
    """[VisualScorePhenotype, GeneInteractionPhenotype] on a fitness record: no label
    value exists, so values [NaN] at type 0, sample 0; the statistic list is
    [gene_interaction_p_value] (visual_score has none) and the statistic tensors are
    empty but present.
    """
    gene = Perturbation().process(
        _small_graph(), VISUAL_THEN_INTERACTION, [_record(["YAL002W"], 0.9, 0.1)]
    )["gene"]
    assert torch.isnan(gene.phenotype_values).tolist() == [True]
    assert gene.phenotype_type_indices.tolist() == [0]
    assert gene.phenotype_sample_indices.tolist() == [0]
    assert gene.phenotype_types == ["visual_score", "gene_interaction"]
    assert gene.phenotype_stat_values.tolist() == []
    assert gene.phenotype_stat_type_indices.tolist() == []
    assert gene.phenotype_stat_sample_indices.tolist() == []
    assert gene.phenotype_stat_types == ["gene_interaction_p_value"]


DCELL_NAMES = ["YAL001C", "YAL002W", "YAL003W", "YAL004W"]


def _named(dcell_graph: HeteroData) -> HeteroData:
    """The conftest DCell graph with gene and term names (it carries no gene ``x``)."""
    dcell_graph["gene"].node_ids = DCELL_NAMES
    dcell_graph["gene_ontology"].node_ids = ["GO:0", "GO:1", "GO:2"]
    return dcell_graph


def test_dcell_on_the_conftest_hierarchy_zeroes_each_perturbed_annotation(
    dcell_graph: HeteroData,
) -> None:
    """Records {YAL003W} (se 0.1) and {YAL001C, YAL003W}: the union is genes {0, 2}, so
    the template rows (1, 0) and (2, 2) get state 0 and (1, 1), (2, 3) keep 1. No
    per-sample ``perturbation_indices_batch`` is written. With [VisualScorePhenotype,
    FitnessPhenotype] fitness sits at type 1 and its se at statistic type 0. The graph
    has no gene ``x``, so none is copied.
    """
    out = DCellGraphProcessor().process(
        _named(dcell_graph),
        VISUAL_THEN_FITNESS,
        [_record(["YAL003W"], 0.5, 0.1), _record(["YAL001C", "YAL003W"], 0.2)],
    )
    gene = out["gene"]
    assert "x" not in gene
    assert gene.perturbation_indices.tolist() == [0, 2]
    assert "perturbation_indices_batch" not in gene
    assert gene.pert_mask.tolist() == [True, False, True, False]
    assert out["gene_ontology"].go_gene_strata_state.tolist() == [
        [1, 0, 1, 0],
        [1, 1, 1, 1],
        [2, 2, 1, 0],
        [2, 3, 1, 1],
    ]
    torch.testing.assert_close(gene.phenotype_values, torch.tensor([0.5, 0.2]))
    assert gene.phenotype_type_indices.tolist() == [1, 1]
    assert gene.phenotype_stat_type_indices.tolist() == [0]
    assert gene.phenotype_stat_sample_indices.tolist() == [0]
    # the template itself is untouched: the processor works on a clone
    assert dcell_graph["gene_ontology"].go_gene_strata_state[:, 3].tolist() == [
        1,
        1,
        1,
        1,
    ]


def test_dcell_with_a_gene_outside_the_graph_collates_through_follow_batch(
    dcell_graph: HeteroData,
) -> None:
    """Contract (issue #527 review): DCell writes no per-sample
    ``perturbation_indices_batch``; ``follow_batch=["perturbation_indices"]`` builds it at
    collate and handles a sample with no perturbed node. YBR001C is not a node:
    perturbation_indices is empty, the state is unchanged, and the name is still listed
    in perturbed_genes. Collating that sample after one perturbing YAL002W (node 1)
    gives perturbation_indices [1] owned by sample 0, i.e. batch vector [0]. A
    per-sample empty key would not collate: PyG increments any key containing "batch"
    by its max, which an empty tensor lacks. The gene_interaction label is absent, so
    values are the [NaN] placeholder and no statistic key is written.
    """
    graph = _named(dcell_graph)
    outside = DCellGraphProcessor().process(
        graph, INTERACTION_ONLY, [_record(["YBR001C"], 0.5)]
    )
    gene = outside["gene"]
    assert gene.perturbed_genes == ["YBR001C"]
    assert gene.perturbation_indices.tolist() == []
    assert "perturbation_indices_batch" not in gene
    assert outside["gene_ontology"].go_gene_strata_state[:, 3].tolist() == [1, 1, 1, 1]
    assert torch.isnan(gene.phenotype_values).tolist() == [True]
    assert not any(key.startswith("phenotype_stat") for key in gene.keys())

    inside = DCellGraphProcessor().process(
        graph, INTERACTION_ONLY, [_record(["YAL002W"], 0.5)]
    )
    batch = Batch.from_data_list(
        [inside, outside], follow_batch=["perturbation_indices"]
    )
    assert batch["gene"].perturbation_indices.tolist() == [1]
    assert batch["gene"].perturbation_indices_batch.tolist() == [0]


def test_dcell_without_a_state_tensor_copies_only_the_term_names(
    dcell_graph: HeteroData,
) -> None:
    """Without ``go_gene_strata_state`` the processor copies only the term names."""
    graph = _named(dcell_graph)
    del graph["gene_ontology"].go_gene_strata_state
    out = DCellGraphProcessor().process(graph, PHENOTYPES, [_record(["YAL002W"], 0.5)])
    assert _plain(out["gene_ontology"]) == {
        "num_nodes": 3,
        "node_ids": ["GO:0", "GO:1", "GO:2"],
    }
    assert "perturbation_indices_batch" not in out["gene"]


def test_neighbor_process_fails_even_without_a_perturbation() -> None:
    """The list-versus-tensor Finding in the subgraph file holds for an empty batch too:
    ``[].tolist()`` raises before any k-hop work.
    """
    with pytest.raises(
        AttributeError, match=r"^'list' object has no attribute 'tolist'$"
    ):
        NeighborSubgraphRepresentation().process(_small_graph(), PHENOTYPES, [])


def test_neighbor_metabolism_stops_at_each_missing_piece() -> None:
    """No reaction node type: nothing is written. Reactions but no gpr: nothing. gpr
    (gene 2 -> r1, gene 0 -> r2) with only gene 1 in the subset: the gpr store is
    written with mask [F, F] and no reaction store. Genes {0, 1}: r2 is reached, the
    reaction store is ['r2'] with its masks, and with no rmr no metabolite store.
    """
    processor = NeighborSubgraphRepresentation(num_hops=1)
    genes_only = HeteroData()
    processor._process_metabolism(
        genes_only, _small_graph(), {"subset_nodes": torch.tensor([0, 1])}
    )
    assert (genes_only.node_types, genes_only.edge_types) == ([], [])

    graph = _small_graph(reactions=True, metabolites=True)
    graph["reaction"].node_ids = ["r0", "r1", "r2"]
    no_gpr = HeteroData()
    processor._process_metabolism(no_gpr, graph, {"subset_nodes": torch.tensor([0])})
    assert (no_gpr.node_types, no_gpr.edge_types) == ([], [])

    graph[GPR].hyperedge_index = torch.tensor([[2, 0], [1, 2]])
    unreached = HeteroData()
    processor._process_metabolism(unreached, graph, {"subset_nodes": torch.tensor([1])})
    assert unreached.node_types == []
    assert unreached[GPR].mask.tolist() == [False, False]

    reached = HeteroData()
    processor._process_metabolism(
        reached, graph, {"subset_nodes": torch.tensor([0, 1])}
    )
    assert reached.node_types == ["reaction"]
    assert _plain(reached["reaction"]) == {
        "node_ids": ["r2"],
        "num_nodes": 1,
        "pert_mask": [False],
        "mask": [True],
    }
    assert reached[GPR].mask.tolist() == [False, True]


def test_neighbor_phenotypes_write_the_shared_placeholder() -> None:
    """Contract (issue #527): with no label value the neighbor processor writes the
    placeholder the subgraph processors write: values [NaN], type [0], sample [0], the
    phenotype_types list, and no statistic keys. With a statistic present the COO
    tensors match the other processors (fitness at type 1 after visual_score, se at
    statistic type 0).
    """
    processor = NeighborSubgraphRepresentation()
    with_stat = HeteroData()
    processor._add_phenotype_data(
        with_stat,
        VISUAL_THEN_FITNESS,
        [_record(["YAL002W"], 0.9, 0.05), _record(["YAL001C"], 0.4)],
    )
    gene = with_stat["gene"]
    torch.testing.assert_close(gene.phenotype_values, torch.tensor([0.9, 0.4]))
    assert gene.phenotype_type_indices.tolist() == [1, 1]
    assert gene.phenotype_sample_indices.tolist() == [0, 1]
    torch.testing.assert_close(gene.phenotype_stat_values, torch.tensor([0.05]))
    assert gene.phenotype_stat_type_indices.tolist() == [0]
    assert gene.phenotype_stat_sample_indices.tolist() == [0]
    assert gene.phenotype_stat_types == ["fitness_se"]

    assert gene.phenotype_types == ["visual_score", "fitness"]

    empty = HeteroData()
    processor._add_phenotype_data(
        empty, INTERACTION_ONLY, [_record(["YAL002W"], 0.9, 0.05)]
    )
    placeholder = empty["gene"]
    assert set(placeholder.keys()) == {
        "phenotype_values",
        "phenotype_type_indices",
        "phenotype_sample_indices",
        "phenotype_types",
    }
    assert torch.isnan(placeholder.phenotype_values).tolist() == [True]
    assert placeholder.phenotype_type_indices.tolist() == [0]
    assert placeholder.phenotype_sample_indices.tolist() == [0]
    assert placeholder.phenotype_types == ["gene_interaction"]


# --- 2026.10.06 (phase 21): the one-line guards ------------------------------------- #

PROCESSORS_WITH_REACTIONS = [
    SubgraphRepresentation,
    IncidenceSubgraphRepresentation,
    LazySubgraphRepresentation,
]


@pytest.mark.parametrize("processor_class", PROCESSORS_WITH_REACTIONS)
def test_empty_reaction_info_and_missing_reactions_write_nothing(
    processor_class: type[Any],
) -> None:
    """``_add_reaction_data`` with ``reaction_info = {}`` returns before reading it, and
    ``_process_metabolic_network`` returns when the cell graph has no reaction node
    type, even with a non-empty reaction_info: the output keeps no node and no edge
    type (graph_processor.py:371/390, 1058/1077, 1720/1740).
    """
    processor = processor_class()
    out = HeteroData()
    processor._add_reaction_data(out, {}, _small_graph())
    processor._process_metabolic_network(out, _small_graph(), {"valid_reactions": [0]})
    processor._process_metabolic_network(out, _small_graph(reactions=True), {})
    assert (out.node_types, out.edge_types) == ([], [])


@pytest.mark.parametrize(
    "processor_class", [*PROCESSORS_WITH_REACTIONS, DCellGraphProcessor]
)
def test_a_device_of_none_falls_back_to_the_cpu(processor_class: type[Any]) -> None:
    """Every processor sets ``device = cpu`` in ``__init__``, so the ``device is None``
    guard of ``_add_phenotype_data`` runs only when a caller clears it: it restores
    the CPU and writes the same phenotype tensors as the default processor.
    """
    records = [_record(["YAL002W"], 0.9, 0.05), _record(["YAL001C"], 0.4)]
    default = HeteroData()
    processor_class()._add_phenotype_data(default, PHENOTYPES, records)
    cleared_processor = processor_class()
    cleared_processor.device = None
    cleared = HeteroData()
    cleared_processor._add_phenotype_data(cleared, PHENOTYPES, records)
    assert cleared_processor.device == torch.device("cpu")
    assert sorted(cleared["gene"].keys()) == sorted(default["gene"].keys())
    for key in default["gene"].keys():
        value = default["gene"][key]
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(cleared["gene"][key], value, equal_nan=True)
        else:
            assert cleared["gene"][key] == value, key
    torch.testing.assert_close(
        default["gene"].phenotype_values, torch.tensor([0.9, 0.4])
    )


def test_the_processor_base_class_cannot_be_instantiated() -> None:
    """``GraphProcessor`` is abstract in ``process``: instantiation is refused."""
    import re

    from torchcell.data.graph_processor import GraphProcessor

    with pytest.raises(
        TypeError,
        match=re.escape(
            "Can't instantiate abstract class GraphProcessor without an implementation "
            "for abstract method 'process'"
        ),
    ):
        GraphProcessor()  # type: ignore[abstract, unused-ignore]
