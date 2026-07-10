"""Unit tests for the sparse ``ExperimentReferenceIndex`` (WS15 streaming redesign).

The reference index is stored as sorted ``member_indices: list[int]`` per reference
(O(N) total) rather than a dense ``list[bool]`` mask per reference (O(references x N)).
These tests pin the representation's invariants -- validation, ``mask()`` reconstruction,
``combine()`` union, and the ``ReferenceIndex`` partition check -- independently of any
particular reference type (the reference is an opaque payload here).
"""

import pytest

from torchcell.data.data import ExperimentReferenceIndex, ReferenceIndex
from torchcell.datamodels.schema import (
    Environment,
    FitnessExperimentReference,
    FitnessPhenotype,
    Media,
    ReferenceGenome,
    Temperature,
)


def _reference(dataset_name: str) -> FitnessExperimentReference:
    """A minimal valid FitnessExperimentReference (payload distinguished by name)."""
    return FitnessExperimentReference(
        dataset_name=dataset_name,
        genome_reference=ReferenceGenome(
            species="Saccharomyces cerevisiae", strain="S288C"
        ),
        environment_reference=Environment(
            media=Media(name="YEPD", state="solid"), temperature=Temperature(value=30.0)
        ),
        phenotype_reference=FitnessPhenotype(
            graph_level="global",
            label_name="fitness",
            label_statistic_name="fitness_std",
            fitness=1.0,
            fitness_std=0.0,
        ),
    )


def test_member_indices_are_sorted_and_deduped() -> None:
    eri = ExperimentReferenceIndex(reference=_reference("d"), member_indices=[5, 1, 3])
    assert eri.member_indices == [1, 3, 5]  # canonical sorted form


def test_member_indices_reject_duplicates_and_negatives() -> None:
    with pytest.raises(ValueError):
        ExperimentReferenceIndex(reference=_reference("d"), member_indices=[1, 1])
    with pytest.raises(ValueError):
        ExperimentReferenceIndex(reference=_reference("d"), member_indices=[-1, 0])


def test_mask_reconstructs_dense_boolean_membership() -> None:
    eri = ExperimentReferenceIndex(reference=_reference("d"), member_indices=[0, 2, 4])
    assert eri.mask(5) == [True, False, True, False, True]
    assert sum(eri.mask(5)) == len(eri.member_indices)


def test_combine_unions_member_indices() -> None:
    ref = _reference("d")
    a = ExperimentReferenceIndex(reference=ref, member_indices=[0, 2])
    b = ExperimentReferenceIndex(reference=ref, member_indices=[2, 3])
    combined = a.combine(b)
    assert combined.member_indices == [0, 2, 3]  # sorted union, no duplicate 2


def test_combine_rejects_different_references() -> None:
    a = ExperimentReferenceIndex(reference=_reference("a"), member_indices=[0])
    b = ExperimentReferenceIndex(reference=_reference("b"), member_indices=[1])
    with pytest.raises(ValueError):
        a.combine(b)


def test_reference_index_partition_accepts_exact_cover() -> None:
    """Two references whose member sets tile range(4) exactly are a valid partition."""
    ri = ReferenceIndex(
        data=[
            ExperimentReferenceIndex(reference=_reference("a"), member_indices=[0, 2]),
            ExperimentReferenceIndex(reference=_reference("b"), member_indices=[1, 3]),
        ]
    )
    assert len(ri) == 2


def test_reference_index_partition_rejects_gap_or_overlap() -> None:
    # overlap: record 1 covered twice, record 2 never
    with pytest.raises(ValueError):
        ReferenceIndex(
            data=[
                ExperimentReferenceIndex(
                    reference=_reference("a"), member_indices=[0, 1]
                ),
                ExperimentReferenceIndex(
                    reference=_reference("b"), member_indices=[1, 3]
                ),
            ]
        )
