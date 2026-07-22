"""Regression tests for GenotypeAggregator.aggregate_check keying.

WS2b changed the aggregator key from the sha256 of
``(systematic_gene_name, perturbation_type)`` tuples to the sha256 of the sorted
SET of systematic gene names ONLY, so that the same genotype measured across
modalities co-locates even when a singleton expression record keeps a
marker-specific perturbation type (e.g. ``kanmx_deletion``) while the replicated
fitness record was rewritten to ``mean_deletion`` by the deduplicator.
"""

import hashlib
from types import SimpleNamespace

from torchcell.data.genotype_aggregate import GenotypeAggregator
from torchcell.datamodels.schema import (
    KanMxDeletionPerturbation,
    MeanDeletionPerturbation,
    NatMxDeletionPerturbation,
    SgaKanMxDeletionPerturbation,
)


def _pert(cls, sys_name, common):
    return cls(systematic_gene_name=sys_name, perturbed_gene_name=common)


def _experiment(perturbations):
    """Wrap perturbations as the minimal shape aggregate_check reads."""
    return {
        "experiment": SimpleNamespace(
            genotype=SimpleNamespace(perturbations=perturbations)
        )
    }


def _key(perturbations):
    return GenotypeAggregator(root="/tmp/unused").aggregate_check(
        _experiment(perturbations)
    )


def test_same_gene_set_different_perturbation_type_share_bucket():
    """The core WS2b fix: marker type must NOT split a genotype's bucket."""
    kanmx = [_pert(KanMxDeletionPerturbation, "YAL001C", "TFC3")]
    mean = [
        MeanDeletionPerturbation(
            systematic_gene_name="YAL001C", perturbed_gene_name="TFC3", num_duplicates=4
        )
    ]
    sga = [
        SgaKanMxDeletionPerturbation(
            systematic_gene_name="YAL001C",
            perturbed_gene_name="TFC3",
            strain_id="test_strain",
        )
    ]
    assert _key(kanmx) == _key(mean) == _key(sga)


def test_different_gene_sets_never_share_bucket():
    """Two genotypes with different gene sets must land in different buckets."""
    a = [_pert(KanMxDeletionPerturbation, "YAL001C", "TFC3")]
    b = [_pert(KanMxDeletionPerturbation, "YAL002W", "VPS8")]
    assert _key(a) != _key(b)


def test_key_is_order_independent():
    """Permuting perturbations (same gene set) yields the same key."""
    p1 = _pert(KanMxDeletionPerturbation, "YAL001C", "TFC3")
    p2 = _pert(NatMxDeletionPerturbation, "YAL002W", "VPS8")
    assert _key([p1, p2]) == _key([p2, p1])


def test_double_vs_single_do_not_collide():
    """A single-KO and a double-KO sharing one gene are distinct buckets."""
    single = [_pert(KanMxDeletionPerturbation, "YAL001C", "TFC3")]
    double = [
        _pert(KanMxDeletionPerturbation, "YAL001C", "TFC3"),
        _pert(NatMxDeletionPerturbation, "YAL002W", "VPS8"),
    ]
    assert _key(single) != _key(double)


def test_key_matches_sorted_gene_set_hash():
    """Key is exactly sha256 of the sorted gene-name set (dedup-compatible)."""
    perts = [
        _pert(KanMxDeletionPerturbation, "YAL002W", "VPS8"),
        MeanDeletionPerturbation(
            systematic_gene_name="YAL001C", perturbed_gene_name="TFC3", num_duplicates=4
        ),
    ]
    expected = hashlib.sha256(str(sorted({"YAL001C", "YAL002W"})).encode()).hexdigest()
    assert _key(perts) == expected
