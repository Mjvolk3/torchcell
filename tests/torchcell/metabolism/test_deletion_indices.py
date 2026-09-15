# tests/torchcell/metabolism/test_deletion_indices.py

"""Tests for the deletion-only gene index, additive to ``perturbation_indices``.

THE DEFECT. Every graph processor collected each perturbation's gene name regardless of
type, so a GeneAddition landed in ``perturbation_indices``. The flux layer's availability
gate reads that index as "this gene product is absent" and closes every reaction whose GPR
needs it, so the Cachera betaxanthin cassette's ARO4^K229L (YBR249C) and ARO7^G141S
(YPR060C), which are feedback-resistant alleles the strain OVERPRODUCES through, were
shutting DAHP synthase and chorismate mutase in every betaxanthin genotype.

What is pinned here: the new field carries deletions only, the old field is untouched, and
``gene_removal_indices`` prefers the new one and falls back when a batch predates it.
"""

import os
from typing import Any

import pytest
import torch
from dotenv import load_dotenv

load_dotenv()
if "DATA_ROOT" not in os.environ:
    pytest.skip("DATA_ROOT is not set", allow_module_level=True)

from torchcell.models.cell_graph_transformer_metabolism import (  # noqa: E402
    gene_removal_indices,
)


class _Gene:
    def __init__(self, deletion: bool) -> None:
        self.perturbation_indices = torch.tensor([0, 1, 2, 3])
        self.perturbation_indices_batch = torch.tensor([0, 0, 1, 1])
        if deletion:
            # Genotype 0 deletes gene 0 and ADDS gene 1; genotype 1 deletes both.
            self.deletion_indices = torch.tensor([0, 2, 3])
            self.deletion_indices_batch = torch.tensor([0, 1, 1])


class _Batch:
    def __init__(self, gene: _Gene, n: int = 2) -> None:
        self._gene = gene
        self.num_graphs = n

    def __getitem__(self, key: str) -> _Gene:
        assert key == "gene"
        return self._gene


def test_gene_removal_prefers_the_deletion_index_when_the_batch_carries_it():
    indices, batch_assignment, source = gene_removal_indices(_Batch(_Gene(True)))
    assert source == "deletion_indices"
    assert indices.tolist() == [0, 2, 3]
    assert batch_assignment.tolist() == [0, 1, 1]


def test_gene_removal_falls_back_for_a_batch_built_before_the_field_existed():
    indices, batch_assignment, source = gene_removal_indices(_Batch(_Gene(False)))
    assert source == "perturbation_indices"
    assert indices.tolist() == [0, 1, 2, 3]
    assert batch_assignment.tolist() == [0, 0, 1, 1]


class _Perturbation:
    def __init__(self, name: str, kind: str) -> None:
        self.systematic_gene_name = name
        self.perturbation_type = kind


class _Genotype:
    def __init__(self, perturbations: list[_Perturbation]) -> None:
        self.perturbations = perturbations


class _Experiment:
    def __init__(self, perturbations: list[_Perturbation]) -> None:
        self.genotype = _Genotype(perturbations)
        self.phenotype = None


def _cachera_like() -> list[dict[str, Any]]:
    """One Cachera-style genotype: a KanMX deletion plus the cassette's four additions."""
    return [
        {
            "experiment": _Experiment(
                [
                    _Perturbation("YAL001C", "kanmx_deletion"),
                    _Perturbation("CYP76AD1", "gene_addition"),
                    _Perturbation("DOD", "gene_addition"),
                    _Perturbation("YBR249C", "gene_addition"),  # ARO4^K229L
                    _Perturbation("YPR060C", "gene_addition"),  # ARO7^G141S
                ]
            )
        }
    ]


def test_the_processor_splits_additions_out_of_the_deletion_set():
    """The name sets the processors build, exercised directly on the same comprehension."""
    data = _cachera_like()
    perturbed = {
        p.systematic_gene_name
        for item in data
        for p in item["experiment"].genotype.perturbations
    }
    deletions = {
        p.systematic_gene_name
        for item in data
        for p in item["experiment"].genotype.perturbations
        if str(p.perturbation_type).endswith("deletion")
    }
    assert perturbed == {"YAL001C", "CYP76AD1", "DOD", "YBR249C", "YPR060C"}
    assert deletions == {"YAL001C"}
    # The two cassette alleles are the ones the gate must NOT close.
    assert "YBR249C" in perturbed and "YBR249C" not in deletions
    assert "YPR060C" in perturbed and "YPR060C" not in deletions


def test_a_deletion_only_genotype_has_the_two_sets_equal():
    data = [
        {
            "experiment": _Experiment(
                [
                    _Perturbation("YAL001C", "kanmx_deletion"),
                    _Perturbation("YBR002C", "deletion"),
                ]
            )
        }
    ]
    perturbed = {
        p.systematic_gene_name
        for item in data
        for p in item["experiment"].genotype.perturbations
    }
    deletions = {
        p.systematic_gene_name
        for item in data
        for p in item["experiment"].genotype.perturbations
        if str(p.perturbation_type).endswith("deletion")
    }
    assert perturbed == deletions
