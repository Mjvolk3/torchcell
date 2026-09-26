# tests/torchcell/data/test_graph_processor.py
# [[tests.torchcell.data.test_graph_processor]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/data/test_graph_processor.py
"""The ``Perturbation`` graph processor on a three-gene cell graph, exact tensors.

Two fitness records perturb {YAL001C} and {YAL002W, YAL003W}. The processor stores the
UNION of perturbed genes for the whole batch, so ``perturbation_indices`` is [0, 1, 2]
and ``pert_mask`` is all True. Phenotypes are COO: one value per record for the
``fitness`` label, type index 0, sample indices 0 and 1; ``fitness_se`` is unset on both
records, so the statistic tensors are empty but the statistic name is still declared.
``tests/torchcell/data/test_graph_processor_equivalence.py`` covers the subgraph
processors on real data behind ``--data``.
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

GENES = GeneSet(["YAL001C", "YAL002W", "YAL003W"])
PHENOTYPES: list[Any] = [FitnessPhenotype]  # the processor takes the classes
ENVIRONMENT = Environment(media=Media(name="YPD", state="solid", is_synthetic=False))


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


def test_perturbation_processor_rejects_an_empty_batch() -> None:
    """No records is an error, not an empty graph."""
    with pytest.raises(ValueError, match="Data list is empty"):
        Perturbation().process(_cell_graph(), PHENOTYPES, [])
