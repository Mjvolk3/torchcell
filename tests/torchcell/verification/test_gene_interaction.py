# tests/torchcell/verification/test_gene_interaction.py
# [[tests.torchcell.verification.test_gene_interaction]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/verification/test_gene_interaction.py
"""The gene-interaction family verifier on hand-built, schema-valid records.

Fixture: three digenic SGA pairs on the shared ``SC`` medium at 26 C, scores -0.2 / 0.1 /
-0.05 with p-values 0.01 / 0.2 / None, each against a reference score 0 with no p-value.
A correct store emits, in order: structural, reference_structural, count,
pair_uniqueness, interaction_order, score_finite, p_value_in_unit_interval,
released_values, reference_zero, reference_environment, signed_unclamped, then the
caller's extra rows, the six shared rules, the two L4 gene rules when a universe is
given, and fitness_companion_containment when companion keys are given.

Derived expectations: the released multiset {(-0.2, 0.01), (0.1, 0.2), (-0.05, NaN)}
sorted by (score, p) is (-0.2, 0.01), (-0.05, NaN), (0.1, 0.2), so swapping the 0.2
p-value for 0.3 changes exactly one sorted position. Two negative and one positive
score give ``n_negative`` 2, ``n_positive`` 1.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from torchcell.datamodels.media import SC
from torchcell.datamodels.schema import (
    Environment,
    GeneInteractionExperiment,
    GeneInteractionExperimentReference,
    GeneInteractionPhenotype,
    Genotype,
    ReferenceGenome,
    SgaKanMxDeletionPerturbation,
    Temperature,
)
from torchcell.verification.gene_interaction import (
    GRAPH_LEVEL_OF_ORDER,
    ReleasedComparison,
    pair_key,
    released_values_result,
    verify_gene_interaction_dataset,
)
from torchcell.verification.released import InteractionValues
from torchcell.verification.report import Level, LevelResult, Provenance

PROV = Provenance(source_uri="test://synthetic", citation_key="kuzminTest2018")
PAIRS = [("YAL001C", "YBR085W"), ("YAL001C", "YJR155W"), ("YBR085W", "YJR155W")]
SCORES = [-0.2, 0.1, -0.05]
P_VALUES: list[float | None] = [0.01, 0.2, None]

FAMILY_NAMES = [
    "structural",
    "reference_structural",
    "count",
    "pair_uniqueness",
    "interaction_order",
    "score_finite",
    "p_value_in_unit_interval",
    "released_values",
    "reference_zero",
    "reference_environment",
    "signed_unclamped",
]
SHARED_NAMES = [
    "provenance_gaps",
    "canonical_gene_names",
    "uncertainty_sanity",
    "compound_identity",
    "media_compound_identity",
    "media_membership",
]


def _record(
    genes: tuple[str, ...],
    score: float,
    p_value: float | None,
    *,
    graph_level: str | None = None,
    reference_score: float = 0.0,
    reference_temperature: float = 26.0,
    screen_id: str | None = None,
) -> dict[str, Any]:
    """One stored ``{experiment, reference}`` gene-interaction record."""
    level = graph_level or GRAPH_LEVEL_OF_ORDER[len(genes)]
    environment = Environment(media=SC, temperature=Temperature(value=26.0))
    experiment = GeneInteractionExperiment(
        dataset_name="test",
        genotype=Genotype(
            perturbations=[
                SgaKanMxDeletionPerturbation(
                    systematic_gene_name=gene,
                    perturbed_gene_name=gene,
                    strain_id=f"{gene}_dma1",
                )
                for gene in genes
            ]
        ),
        environment=environment,
        phenotype=GeneInteractionPhenotype(
            graph_level=level,
            gene_interaction=score,
            gene_interaction_p_value=p_value,
            screen_id=screen_id,
        ),
    )
    reference = GeneInteractionExperimentReference(
        dataset_name="test",
        genome_reference=ReferenceGenome(
            species="Saccharomyces cerevisiae", strain="S288C"
        ),
        environment_reference=Environment(
            media=SC, temperature=Temperature(value=reference_temperature)
        ),
        phenotype_reference=GeneInteractionPhenotype(
            graph_level=level, gene_interaction=reference_score
        ),
    )
    return {"experiment": experiment.model_dump(), "reference": reference.model_dump()}


def _records() -> list[dict[str, Any]]:
    return [_record(g, s, p) for g, s, p in zip(PAIRS, SCORES, P_VALUES)]


def _values(scores: list[float], p_values: list[float | None]) -> InteractionValues:
    return InteractionValues(
        scores=np.array(scores, dtype=np.float64),
        p_values=np.array(
            [math.nan if p is None else p for p in p_values], dtype=np.float64
        ),
    )


def _released(
    scores: list[float] | None = None,
    p_values: list[float | None] | None = None,
    *,
    drift: dict[str, str] | None = None,
) -> ReleasedComparison:
    return ReleasedComparison(
        files=["data_s1.tsv"],
        drift=drift or {},
        values=None
        if drift
        else _values(scores or SCORES, P_VALUES if p_values is None else p_values),
        description="the digenic rows",
    )


def _verify(records: list[dict[str, Any]], **kwargs: Any) -> Any:
    arguments: dict[str, Any] = {
        "dataset_name": "dmi",
        "provenance": PROV,
        "expected_count": len(records),
        "order": 2,
        "released": _released(),
        "companion_keys": None,
        "companion_name": None,
    }
    arguments.update(kwargs)
    return verify_gene_interaction_dataset(records, **arguments)


def _result(report: Any, name: str) -> LevelResult:
    matches = [r for r in report.results if r.name == name]
    assert len(matches) == 1, [r.name for r in report.results]
    result: LevelResult = matches[0]
    return result


def test_a_correct_digenic_store_emits_every_row_in_order_and_passes() -> None:
    report = _verify(_records())
    assert [r.name for r in report.results] == FAMILY_NAMES + SHARED_NAMES
    assert [r.level for r in report.results[: len(FAMILY_NAMES)]] == [
        Level.L0,
        Level.L0,
        Level.L1,
        Level.L1,
        Level.L1,
        Level.L2,
        Level.L2,
        Level.L2,
        Level.L3,
        Level.L3,
        Level.L3,
    ]
    assert report.passed is True
    assert _result(report, "structural").message == (
        "3 records validated as GeneInteractionExperiment"
    )
    assert _result(report, "interaction_order").message == (
        "all 3 records perturb 2 distinct genes at graph level 'edge'"
    )
    assert _result(report, "p_value_in_unit_interval").message == (
        "2 stated p-values lie in [0, 1] (1 records state none)"
    )
    assert _result(report, "released_values").details["n_differing_positions"] == 0
    assert _result(report, "signed_unclamped").details == {
        "n_negative": 2,
        "n_positive": 1,
        "n_zero": 0,
    }


def test_extra_rows_follow_the_family_rows_and_gene_rules_follow_the_shared_rows() -> (
    None
):
    extra = LevelResult(
        level=Level.L3, name="provenance_audit", passed=True, message="m"
    )
    report = _verify(
        _records(),
        extra_results=[extra],
        sgd_genes={"YAL001C", "YBR085W", "YJR155W"},
        companion_keys={pair_key(r["experiment"]) for r in _records()},
        companion_name="dmf",
    )
    assert [r.name for r in report.results] == (
        FAMILY_NAMES
        + ["provenance_audit"]
        + SHARED_NAMES
        + ["gene_containment_sgd", "current_genome_genes"]
        + ["fitness_companion_containment"]
    )
    assert report.passed is True
    assert _result(report, "fitness_companion_containment").message == (
        "every one of the 3 (strain, environment) keys is a record of dmf"
    )


def test_a_key_missing_from_the_companion_fails_containment() -> None:
    records = _records()
    report = _verify(
        records,
        companion_keys={pair_key(records[0]["experiment"])},
        companion_name="dmf",
    )
    result = _result(report, "fitness_companion_containment")
    assert result.passed is False
    assert result.level is Level.L4
    assert result.details == {
        "companion": "dmf",
        "n_keys": 3,
        "n_companion_keys": 1,
        "n_missing": 2,
    }


def test_two_records_of_one_strain_and_environment_are_a_duplicate_unless_screens_differ() -> (
    None
):
    twice = [_record(PAIRS[0], -0.2, 0.01), _record(PAIRS[0], -0.1, 0.02)]
    report = _verify(twice, released=_released([-0.2, -0.1], [0.01, 0.02]))
    assert _result(report, "pair_uniqueness").details == {
        "n_pairs": 1,
        "n_duplicated": 1,
        "n_extra_records": 1,
    }
    screens = [
        _record(PAIRS[0], -0.2, 0.01, screen_id="s1"),
        _record(PAIRS[0], -0.1, 0.02, screen_id="s3"),
    ]
    report = _verify(screens, released=_released([-0.2, -0.1], [0.01, 0.02]))
    assert _result(report, "pair_uniqueness").passed is True


def test_interaction_order_names_the_wrong_arity_gene_count_and_level() -> None:
    records = [
        _record(("YAL001C", "YBR085W", "YJR155W"), -0.2, 0.01, graph_level="edge"),
        _record(("YAL001C", "YAL001C"), 0.1, 0.2),
        _record(("YAL001C", "YBR085W"), -0.05, None, graph_level="hyperedge"),
    ]
    result = _result(_verify(records), "interaction_order")
    assert result.passed is False
    assert result.details["by_problem"] == {
        "3 perturbations": 1,
        "1 distinct genes": 1,
        "graph_level 'hyperedge'": 1,
    }
    trigenic = _verify(
        [_record(("YAL001C", "YBR085W", "YJR155W"), -0.2, 0.01)],
        order=3,
        released=_released([-0.2], [0.01]),
    )
    assert _result(trigenic, "interaction_order").message == (
        "all 1 records perturb 3 distinct genes at graph level 'hyperedge'"
    )


def test_an_unknown_order_is_refused_before_any_record_is_read() -> None:
    with pytest.raises(ValueError, match="interaction order 4 is not 2 or 3"):
        _verify(_records(), order=4)


def test_non_finite_scores_and_out_of_range_p_values_fail_their_rows() -> None:
    records = _records()
    records[0]["experiment"]["phenotype"]["gene_interaction"] = math.inf
    records[1]["experiment"]["phenotype"]["gene_interaction_p_value"] = 1.5
    report = _verify(records)
    finite = _result(report, "score_finite")
    assert finite.passed is False
    assert finite.details["bad"] == [{"index": 0, "value": "inf"}]
    p_rule = _result(report, "p_value_in_unit_interval")
    assert p_rule.passed is False
    assert p_rule.details["bad"] == [{"index": 1, "value": "1.5"}]
    assert _result(report, "signed_unclamped").details["n_negative"] == 1


def test_a_non_numeric_score_is_a_bad_score_and_a_nan_in_the_stored_multiset() -> None:
    records = _records()
    records[2]["experiment"]["phenotype"]["gene_interaction"] = "x"
    report = _verify(records)
    assert _result(report, "score_finite").details["bad"] == [
        {"index": 2, "value": "'x'"}
    ]
    assert _result(report, "released_values").passed is False
    assert _result(report, "structural").passed is False


def test_released_values_fail_on_drift_on_size_and_on_a_changed_value() -> None:
    drift = _result(
        _verify(_records(), released=_released(drift={"data_s1.tsv": "abc"})),
        "released_values",
    )
    assert drift.passed is False
    assert drift.message == (
        "sha256 drift in ['data_s1.tsv']: the release was not read"
    )
    short = _result(
        _verify(_records(), released=_released([-0.2, 0.1], [0.01, 0.2])),
        "released_values",
    )
    assert short.message == (
        "3 stored scores against 2 released rows (the digenic rows)"
    )
    changed = _result(
        _verify(_records(), released=_released(SCORES, [0.01, 0.3, None])),
        "released_values",
    )
    assert changed.passed is False
    assert changed.details["n_differing_positions"] == 1
    assert changed.details["examples"] == [
        {
            "stored_score": 0.1,
            "stored_p_value": 0.2,
            "released_score": 0.1,
            "released_p_value": 0.3,
        }
    ]


def test_released_comparison_without_drift_must_carry_values() -> None:
    with pytest.raises(ValueError, match="must carry its values"):
        ReleasedComparison(files=["f"], drift={}, values=None, description="d")


def test_released_values_result_compares_the_multiset_not_the_order() -> None:
    stored = _values([0.1, -0.2], [0.2, None])
    released = ReleasedComparison(
        files=["f"],
        drift={},
        values=_values([-0.2, 0.1], [None, 0.2]),
        description="rows",
    )
    result = released_values_result(stored, released)
    assert result.passed is True
    assert result.message == (
        "the 2 stored (score, p-value) pairs are the released multiset (rows)"
    )


def test_reference_rows_catch_a_nonzero_reference_and_another_environment() -> None:
    records = [
        _record(PAIRS[0], -0.2, 0.01, reference_score=0.5),
        _record(PAIRS[1], 0.1, 0.2, reference_temperature=30.0),
        _record(PAIRS[2], -0.05, None),
    ]
    records[2]["reference"]["phenotype_reference"]["gene_interaction_p_value"] = 0.4
    report = _verify(records)
    zero = _result(report, "reference_zero")
    assert zero.passed is False
    assert zero.details == {
        "reference_scores": {"0.0": 2, "0.5": 1},
        "n_reference_with_p_value": 1,
        "n_reference_wrong_graph_level": 0,
    }
    environment = _result(report, "reference_environment")
    assert environment.passed is False
    assert environment.details == {"n_differ": 1, "example_indices": [1]}


def test_one_signed_store_fails_signed_unclamped_and_a_zero_counts_as_zero() -> None:
    records = [
        _record(PAIRS[0], 0.2, 0.01),
        _record(PAIRS[1], 0.0, 0.5),
        _record(PAIRS[2], 0.1, None),
    ]
    report = _verify(records, released=_released([0.2, 0.0, 0.1], [0.01, 0.5, None]))
    signed = _result(report, "signed_unclamped")
    assert signed.passed is False
    assert signed.message == "0 aggravating and 2 alleviating scores, 1 at exactly zero"


def test_an_empty_store_fails_every_row_that_needs_a_record() -> None:
    report = _verify(
        [],
        expected_count=0,
        released=_released([], []),
        companion_keys=set(),
        companion_name="dmf",
    )
    failed = {r.name for r in report.results if not r.passed}
    assert {
        "structural",
        "reference_structural",
        "pair_uniqueness",
        "interaction_order",
        "reference_zero",
        "reference_environment",
        "signed_unclamped",
        "fitness_companion_containment",
    } <= failed


def test_a_record_that_is_not_its_declared_class_fails_structural() -> None:
    records = _records()
    records[0]["experiment"]["experiment_type"] = "fitness"
    records[1]["reference"]["experiment_reference_type"] = "fitness"
    report = _verify(records)
    structural = _result(report, "structural")
    assert structural.passed is False
    assert structural.details["n_failures"] == 1
    assert (
        "experiment_type 'fitness' is not GeneInteractionExperiment's"
        in (structural.details["failures"][0]["error"])
    )
    assert _result(report, "reference_structural").details["n_failures"] == 1


def test_pair_key_is_the_same_for_two_stores_built_from_the_same_strain() -> None:
    first = _record(PAIRS[0], -0.2, 0.01)["experiment"]
    second = _record(PAIRS[0], 0.4, 0.9)["experiment"]
    assert pair_key(first) == pair_key(second)
    assert len(pair_key(first)) == 16
    assert pair_key(first) != pair_key(_record(PAIRS[1], -0.2, 0.01)["experiment"])
