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
"""

from typing import Any

import networkx as nx
import pytest
import torch
from sortedcontainers import SortedDict
from torch_geometric.data import HeteroData

from torchcell.data.cell_data import to_cell_data
from torchcell.data.graph_processor import Perturbation
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
)
from torchcell.graph.graph import GeneGraph, GeneMultiGraph
from torchcell.sequence import GeneSet

GENES = GeneSet(["YAL001C", "YAL002W", "YAL003W"])
PHENOTYPES: list[Any] = [FitnessPhenotype]  # the processor takes the classes
TWO_PHENOTYPES: list[Any] = [FitnessPhenotype, GeneInteractionPhenotype]
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
