# tests/torchcell/verification/test_pooled_sort_seq.py
# [[tests.torchcell.verification.test_pooled_sort_seq]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/verification/test_pooled_sort_seq.py
"""The pooled sort-seq verifier's levels, each on the record set it is about (#991).

The family's identity rule is the SEQUENCE, not a well, so the levels that matter are the
ones an arrayed reporter library has no equivalent of: one record per ``sequence_id``, a
POOLED readout member, the record's own sequence on its genotype, and a typed gap wherever
the release carries no dispersion. The end-to-end run over a built store is in
``tests/torchcell/datasets/scerevisiae/test_vaishnav2022.py``; this module holds each level
against hand-built records, so a failure names the rule rather than the dataset.
"""

from __future__ import annotations

from typing import Any

from torchcell.verification import pooled_sort_seq as psq

SEQUENCE = "TGCATTTTTTTCACATC" + "A" * 80 + "GGTTACGGCTGTT"


def _record(**phenotype: Any) -> dict[str, Any]:
    """One record shaped like a stored one, with the phenotype fields overridden."""
    base: dict[str, Any] = {
        "expression_level": 8.5,
        "expression_units": "sort-bin index",
        "sequence_id": "A",
        "readout": "sort_seq_fluorescence",
        "reporter_gene": "YFP",
        "read_count": None,
        "provenance_gaps": [
            {
                "field": "expression_level_uncertainty",
                "reason": "not_reported_by_primary",
            },
            {
                "field": "expression_level_uncertainty_type",
                "reason": "not_reported_by_primary",
            },
        ],
    }
    base.update(phenotype)
    return {
        "experiment": {
            "phenotype": base,
            "genotype": {
                "perturbations": [
                    {
                        "promoter_variant_id": base["sequence_id"],
                        "promoter_sequence": SEQUENCE,
                        "variable_region_start": 18,
                        "variable_region_end": 97,
                    }
                ]
            },
        }
    }


def test_sequence_uniqueness_is_the_pooled_identity_rule() -> None:
    """One record per sequence id; a repeat is one measurement stored twice."""
    passing = psq._l1_sequence_uniqueness(["A", "B", "C"])
    assert passing.passed
    assert passing.details == {"n_records": 3, "n_sequences": 3, "duplicates": []}
    failing = psq._l1_sequence_uniqueness(["A", "B", "A"])
    assert not failing.passed
    assert failing.details == {"n_records": 3, "n_sequences": 2, "duplicates": ["A"]}


def test_units_declared_accepts_the_pooled_member_and_counts_the_readouts() -> None:
    """The pooled readout passes and the census of readouts is in the report."""
    result = psq._l3_units_declared([_record(), _record(sequence_id="B")])
    assert result.passed
    assert result.details["readouts"] == {"sort_seq_fluorescence": 2}
    assert result.details["not_pooled"] == {}


def test_units_declared_refuses_a_per_well_readout() -> None:
    """A plate-reader member would claim a sample unit a pooled estimate lacks."""
    result = psq._l3_units_declared([_record(readout="plate_reader_fluorescence")])
    assert not result.passed
    assert result.details["not_pooled"] == {"plate_reader_fluorescence": 1}


def test_units_declared_refuses_a_record_with_no_units() -> None:
    """Units, reporter and readout are all required for a number to be comparable."""
    result = psq._l3_units_declared([_record(expression_units="  ")])
    assert not result.passed
    assert result.details["n_missing"] == 1


def test_sequence_is_the_genotype_passes_and_counts_the_lengths() -> None:
    """The genotype carries the record's own sequence, and its length is reported."""
    result = psq._l3_sequence_is_the_genotype([_record()])
    assert result.passed
    assert result.details["sequence_lengths"] == {110: 1}


def test_sequence_is_the_genotype_refuses_a_mismatched_variant_id() -> None:
    """A genotype naming another variant would key the number to the wrong sequence."""
    record = _record()
    record["experiment"]["genotype"]["perturbations"][0]["promoter_variant_id"] = "Z"
    result = psq._l3_sequence_is_the_genotype([record])
    assert not result.passed
    assert result.details["problems"] == [
        {
            "sequence_id": "A",
            "why": "the genotype's variant id is not the phenotype's sequence id",
        }
    ]


def test_sequence_is_the_genotype_refuses_a_degenerate_sequence_and_a_bad_span() -> (
    None
):
    """An ambiguous base or a span outside the sequence is named, not stored."""
    degenerate = _record()
    degenerate["experiment"]["genotype"]["perturbations"][0]["promoter_sequence"] = (
        "ACGN"
    )
    assert psq._l3_sequence_is_the_genotype([degenerate]).details["problems"] == [
        {"sequence_id": "A", "why": "the promoter sequence is not unambiguous ACGT"}
    ]
    span = _record()
    span["experiment"]["genotype"]["perturbations"][0]["variable_region_end"] = 400
    assert psq._l3_sequence_is_the_genotype([span]).details["problems"] == [
        {
            "sequence_id": "A",
            "why": "the variable span 18-400 is not inside the sequence",
        }
    ]


def test_sequence_is_the_genotype_refuses_two_promoter_bearing_perturbations() -> None:
    """One construct per record: two would make the measured sequence ambiguous."""
    record = _record()
    perturbations = record["experiment"]["genotype"]["perturbations"]
    perturbations.append(dict(perturbations[0]))
    result = psq._l3_sequence_is_the_genotype([record])
    assert not result.passed
    assert result.details["problems"] == [
        {"sequence_id": "A", "why": "2 promoter-bearing perturbations"}
    ]


def test_uncertainty_is_typed_or_gapped_counts_both_honest_forms() -> None:
    """A typed uncertainty and a typed gap both pass; a silent None does not."""
    typed = _record(
        expression_level_uncertainty=1.0,
        expression_level_uncertainty_type="sample_sd",
        provenance_gaps=[],
    )
    result = psq._l3_uncertainty_is_typed_or_gapped([typed, _record(sequence_id="B")])
    assert result.passed
    assert (result.details["n_typed"], result.details["n_gapped"]) == (1, 1)
    silent = psq._l3_uncertainty_is_typed_or_gapped([_record(provenance_gaps=[])])
    assert not silent.passed
    assert silent.details["silent"] == ["A"]


def test_released_values_compares_the_store_with_the_release() -> None:
    """L2 is the bit-for-bit check that the loader transformed nothing."""
    passing = psq.l2_released_values([_record()], {"A": 8.5})
    assert passing.passed
    assert passing.details["max_abs_delta"] == 0.0
    drifted = psq.l2_released_values([_record(expression_level=9.0)], {"A": 8.5})
    assert not drifted.passed
    assert drifted.details["drifted"] == [{"sequence_id": "A", "delta": 0.5}]
    absent = psq.l2_released_values([_record(sequence_id="Z")], {"A": 8.5})
    assert not absent.passed
    assert absent.details["not_in_release"] == ["Z"]


def test_read_count_coverage_reports_rather_than_requires() -> None:
    """A release that carries no read count is honest; the count belongs in the report."""
    none = psq.l4_read_count_coverage([_record(), _record(sequence_id="B")])
    assert none.passed
    assert none.details == {"n_records": 2, "n_with_read_count": 0, "smallest": None}
    some = psq.l4_read_count_coverage(
        [_record(read_count=647), _record(sequence_id="B", read_count=12)]
    )
    assert some.passed
    assert some.details == {"n_records": 2, "n_with_read_count": 2, "smallest": 12}


def test_the_pooled_readout_set_is_the_sort_seq_member_alone() -> None:
    """The member the schema added for this family, pinned against a per-well one."""
    assert psq.POOLED_READOUTS == frozenset({"sort_seq_fluorescence"})


def test_estimate_is_stated_or_gapped_counts_the_releases_own_absence() -> None:
    """A release row with no estimate is a record with a typed gap on the label."""
    gapped = _record(
        expression_level=None,
        provenance_gaps=[
            {"field": "expression_level", "reason": "not_reported_by_primary"}
        ],
    )
    result = psq._l3_estimate_is_stated_or_gapped([_record(), gapped])
    assert result.passed
    assert (result.details["n_stated"], result.details["n_gapped"]) == (1, 1)


def test_estimate_is_stated_or_gapped_refuses_a_silent_absence_and_a_gapped_value() -> (
    None
):
    """Neither a missing level with no gap nor a stated level with one is honest."""
    silent = psq._l3_estimate_is_stated_or_gapped(
        [_record(expression_level=None, provenance_gaps=[])]
    )
    assert not silent.passed
    assert silent.details["problems"] == [
        {"sequence_id": "A", "why": "an absent estimate carries no gap"}
    ]
    both = psq._l3_estimate_is_stated_or_gapped(
        [
            _record(
                provenance_gaps=[
                    {"field": "expression_level", "reason": "not_reported_by_primary"}
                ]
            )
        ]
    )
    assert not both.passed
    assert both.details["problems"] == [
        {"sequence_id": "A", "why": "a stated estimate also carries a gap"}
    ]


def test_released_values_agrees_on_an_absence_and_refuses_a_disagreement() -> None:
    """The release decides: an absence it states must be an absence in the store."""
    absent = _record(expression_level=None)
    agree = psq.l2_released_values([absent], {"A": None})
    assert agree.passed
    assert agree.details["n_absent_agree"] == 1
    stored_anyway = psq.l2_released_values([_record()], {"A": None})
    assert not stored_anyway.passed
    assert stored_anyway.details["drifted"] == [
        {
            "sequence_id": "A",
            "why": "the release states no estimate and the record does",
        }
    ]
    dropped = psq.l2_released_values([absent], {"A": 8.5})
    assert not dropped.passed
    assert dropped.details["drifted"] == [
        {
            "sequence_id": "A",
            "why": "the release states an estimate and the record does not",
        }
    ]
