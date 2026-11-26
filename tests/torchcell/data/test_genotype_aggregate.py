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

from torchcell.data.genotype_aggregate import (
    GenotypeAggregator,
    ReferenceKeyedGenotypeAggregator,
)
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


# --- ReferenceKeyedGenotypeAggregator -----------------------------------------
#
# Added with experiment 048. The class exists because a genotype-keyed
# aggregator merges two measurements of one design taken in different screens,
# and in a design-build-test-learn campaign that is the contrast the campaign
# was run to produce. The key is (perturbation_type, gene) pairs together with
# the reference phenotype's own value, and both of those halves are pinned here.


def _reference(titer):
    """A reference carrying one labeled control value, as the key reads it."""
    return SimpleNamespace(
        phenotype_reference=SimpleNamespace(label_name="titer", titer=titer)
    )


def _ref_pair(perturbations, titer):
    """The pydantic-shaped pair ``aggregate_check`` reads."""
    return {
        "experiment": SimpleNamespace(
            genotype=SimpleNamespace(perturbations=perturbations)
        ),
        "experiment_reference": _reference(titer),
    }


def _raw_pair(perturbations, titer):
    """The raw-dict-shaped record ``aggregate_key_raw`` reads."""
    return {
        "experiment": {
            "genotype": {
                "perturbations": [
                    {"systematic_gene_name": name, "perturbation_type": kind}
                    for name, kind in perturbations
                ]
            }
        },
        "experiment_reference": {
            "phenotype_reference": {"label_name": "titer", "titer": titer}
        },
    }


def _ref_key(perturbations, titer):
    return ReferenceKeyedGenotypeAggregator(root="/tmp/unused").aggregate_check(
        _ref_pair(perturbations, titer)
    )


def _raw_key(perturbations, titer):
    return ReferenceKeyedGenotypeAggregator(root="/tmp/unused").aggregate_key_raw(
        _raw_pair(perturbations, titer)
    )


def test_reference_value_separates_the_same_design():
    """One design measured against two controls is two samples, not one."""
    perturbations = [_pert(KanMxDeletionPerturbation, "YAL001C", "TFC3")]
    assert _ref_key(perturbations, 156.7333) != _ref_key(perturbations, 168.8314)


def test_same_design_and_same_reference_share_a_bucket():
    """Replicates of one design against one control still aggregate."""
    perturbations = [_pert(KanMxDeletionPerturbation, "YAL001C", "TFC3")]
    assert _ref_key(perturbations, 156.7333) == _ref_key(perturbations, 156.7333)


def test_perturbation_type_splits_what_the_gene_set_key_merges():
    """The axis-aware half: one gene, two mechanisms, two genotypes.

    ``GenotypeAggregator`` keys on gene names alone and gives these two the same
    bucket, which is the TODO the new class closes.
    """
    kanmx = [_pert(KanMxDeletionPerturbation, "YAL001C", "TFC3")]
    natmx = [_pert(NatMxDeletionPerturbation, "YAL001C", "TFC3")]
    assert _key(kanmx) == _key(natmx)
    assert _ref_key(kanmx, 100.0) != _ref_key(natmx, 100.0)


def test_both_key_paths_agree_on_the_same_record():
    """The base class requires it: the raw path must match the pydantic path."""
    pairs = [("YAL001C", "kanmx_deletion"), ("YBR002W", "kanmx_deletion")]
    perturbations = [
        _pert(KanMxDeletionPerturbation, "YAL001C", "TFC3"),
        _pert(KanMxDeletionPerturbation, "YBR002W", "RER2"),
    ]
    assert _ref_key(perturbations, 156.7333) == _raw_key(pairs, 156.7333)


def test_an_integral_reference_value_keys_as_its_float():
    """A JSON integer and a validated float are the same measurement.

    pydantic coerces an integral value in a ``float`` field to ``100.0`` while
    the stored JSON keeps ``100``. Canonicalizing through ``float`` is what keeps
    the two key paths from splitting one record into two buckets.
    """
    pairs = [("YAL001C", "kanmx_deletion")]
    perturbations = [_pert(KanMxDeletionPerturbation, "YAL001C", "TFC3")]
    assert _raw_key(pairs, 100) == _raw_key(pairs, 100.0)
    assert _ref_key(perturbations, 100.0) == _raw_key(pairs, 100)


def test_a_missing_reference_value_is_its_own_key():
    """A typed provenance gap degrades to one bucket rather than raising."""
    pairs = [("YAL001C", "kanmx_deletion")]
    assert _raw_key(pairs, None) == _raw_key(pairs, None)
    assert _raw_key(pairs, None) != _raw_key(pairs, 100.0)


def test_key_is_order_independent_in_both_halves():
    """Neither the perturbation order nor the pair order changes the key."""
    forward = [("YAL001C", "kanmx_deletion"), ("YBR002W", "natmx_deletion")]
    backward = list(reversed(forward))
    assert _raw_key(forward, 156.7333) == _raw_key(backward, 156.7333)
