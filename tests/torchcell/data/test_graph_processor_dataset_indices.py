"""Perturbation processor: the per-entry source-dataset index beside each value."""

from typing import Any

import pytest
import torch
from torch_geometric.data import Batch, HeteroData

from torchcell.data.graph_processor import Perturbation
from torchcell.datamodels.schema import (
    Environment,
    FitnessExperiment,
    FitnessPhenotype,
    Genotype,
    Media,
    SgaKanMxDeletionPerturbation,
    Temperature,
)

GENES = ["YAL001C", "YAL002W", "YAL003W", "YAL004W"]
PHENOTYPES: list[Any] = [FitnessPhenotype]


def _cell_graph() -> HeteroData:
    cg = HeteroData()
    cg["gene"].num_nodes = len(GENES)
    cg["gene"].node_ids = list(GENES)
    return cg


def _entry(gene: str, dataset: str, fitness: float) -> dict[str, Any]:
    env = Environment(
        media=Media(name="YEPD", state="solid", is_synthetic=False),
        temperature=Temperature(value=30.0),
    )
    genotype = Genotype(
        perturbations=[
            SgaKanMxDeletionPerturbation(
                systematic_gene_name=gene,
                perturbed_gene_name=gene,
                strain_id=f"{gene}_dma1",
            )
        ]
    )
    return {
        "experiment": FitnessExperiment(
            dataset_name=dataset,
            genotype=genotype,
            environment=env,
            phenotype=FitnessPhenotype(fitness=fitness, fitness_std=None),
        ),
        # The processor reads the experiment only; the reference is not touched.
        "experiment_reference": None,
    }


def test_indices_follow_the_vocabulary_and_pair_with_every_value() -> None:
    vocab = ["SmfCostanzo2016Dataset", "SmfKuzmin2018Dataset"]
    proc = Perturbation(dataset_vocabulary=vocab)
    data = [
        _entry("YAL001C", "SmfKuzmin2018Dataset", 0.9),
        _entry("YAL001C", "SmfCostanzo2016Dataset", 0.8),
        _entry("YAL001C", "SmfCostanzo2016Dataset", 0.7),
    ]
    out = proc.process(_cell_graph(), PHENOTYPES, data)
    g = out["gene"]
    assert g.phenotype_values.tolist() == pytest.approx([0.9, 0.8, 0.7])
    assert g.phenotype_dataset_indices.tolist() == [1, 0, 0]
    assert g.phenotype_dataset_indices.shape == g.phenotype_values.shape
    assert g.phenotype_dataset_indices.dtype == torch.long


def test_unknown_dataset_raises_and_no_vocabulary_emits_nothing() -> None:
    with pytest.raises(ValueError, match="not in the dataset vocabulary"):
        Perturbation(dataset_vocabulary=["SmfCostanzo2016Dataset"]).process(
            _cell_graph(), PHENOTYPES, [_entry("YAL001C", "SmfKuzmin2018Dataset", 0.9)]
        )
    out = Perturbation().process(
        _cell_graph(), PHENOTYPES, [_entry("YAL001C", "SmfKuzmin2018Dataset", 0.9)]
    )
    assert not hasattr(out["gene"], "phenotype_dataset_indices")


def test_collate_keeps_the_indices_unshifted_and_aligned_with_values_batch() -> None:
    vocab = ["SmfCostanzo2016Dataset", "SmfKuzmin2018Dataset"]
    proc = Perturbation(dataset_vocabulary=vocab)
    a = proc.process(
        _cell_graph(),
        PHENOTYPES,
        [
            _entry("YAL001C", "SmfKuzmin2018Dataset", 0.9),
            _entry("YAL001C", "SmfCostanzo2016Dataset", 0.8),
        ],
    )
    b = proc.process(
        _cell_graph(), PHENOTYPES, [_entry("YAL002W", "SmfCostanzo2016Dataset", 0.5)]
    )
    batch = Batch.from_data_list(
        [a, b], follow_batch=["perturbation_indices", "phenotype_values"]
    )
    g = batch["gene"]
    # No num_nodes offset (the key has no "index" substring), and one row per value.
    assert g.phenotype_dataset_indices.tolist() == [1, 0, 0]
    assert g.phenotype_values_batch.tolist() == [0, 0, 1]
